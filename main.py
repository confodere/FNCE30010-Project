import re, json
from datetime import datetime
from System.Collections.Generic import KeyNotFoundException

from bs4 import BeautifulSoup
import bs4
import pandas as pd
import numpy as np

from QuantConnect.DataSource import SECReport8K
from AlgorithmImports import *


def decode(string):
    unsafe_doc = string.Report.Documents[0].Text
    doc = bytes(unsafe_doc, "utf-8").decode("unicode_escape")
    return doc


def identify(report):
    doc = decode(report)
    search_str = "Annual Meeting of Shareholders"
    or_strs = ["Annual Meeting", "Submission of Matters to a Vote"]
    return search_str in doc or (or_strs[0] in doc and or_strs[1] in doc)


FALSE_POSITIVES = {
    "UNH": ["2015-03-30", "2015-07-01"],
    "NKE": ["2010-09-24", "2014-04-21", "2014-06-23", "2015-02-12", "2015-06-30", "2019-04-23", "2019-11-14"],
    "PG": ["2010-04-22", "2013-12-13", "2015-06-12", "2016-04-12"],
    "AMZN": ["2010-03-16"],
    "HD": ["2011-03-24", "2019-03-04"],
    "JPM": ["2012-04-04", "2013-06-10"],
}
BENCHMARK_RATE = 70
BM = False


class AlgoProject(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)  # Set Start Date
        self.SetEndDate(2019, 12, 31)
        self.SetCash(100000)  # Set Strategy Cash

        self._changes = None
        self._symbols = {}
        self._logged_portfolio = {}

        pd.set_option("display.max_rows", 50)
        pd.set_option("display.max_columns", 30)
        pd.set_option("display.width", 1000)

        self.UniverseSettings.Resolution = Resolution.Daily
        # self.AddUniverse(self.Universe.QC500)

        """self._universe = [
            "AAPL",
            "MSFT",
            "GOOG",
            "AMZN",
            "TSLA" "FB",
            "BRK.A",
            "NVDA",
            "TSM",
            "JPM",
            "V",
            "BABA",
            "JNJ",
            "UNH",
            "WMT",
            "BAC",
            "HD",
            "MA",
            "PG",
            "ASML",
            "DIS",
            "ADBE",
            "NFLX",
            "CRMPYPL",
            "ORCL",
            "XOM",
            "NKE",
        ]"""
        self.AddEquity("SPY")

        self._universe = ["GOOG", "AAPL", "ADBE", "AMZN", "JPM", "MA", "MSFT", "NFLX", "ORCL", "PG"]
        # self._universe = ["AAPL", "GOOG"]
        self._asset_no = len(self._universe)

        self._voteData = {}

        for equity in self._universe:
            security = self.AddEquity(equity, Resolution.Daily)
            self._voteData[equity] = MeetingContainer(equity)

        self.SetWarmUp(1)

        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.AfterMarketOpen("SPY"), self.AlgDynamicWeight)

        self.Schedule.On(self.DateRules.MonthEnd(), self.TimeRules.At(12, 0), self._checkPortfolio)

    def AlgThreshold(self):

        targets = []
        divest = []
        for equity, container in self._voteData.items():
            year = self.Time.year
            if hasattr(container, str(year)) and getattr(container, str(year)).PassRate > BENCHMARK_RATE:
                targets.append(container.symbol)
            elif hasattr(container, str(year - 1)) and getattr(container, str(year - 1)).PassRate > BENCHMARK_RATE:
                targets.append(container.symbol)
            else:
                if BM:
                    targets.append(container.symbol)
                else:
                    divest.append(container.symbol)

        target_list = [PortfolioTarget(equity, 1 / len(targets)) for equity in targets]
        divest_list = [PortfolioTarget(equity, 0) for equity in divest]
        self.SetHoldings(divest_list + target_list)

    def AlgDynamicWeight(self):

        approval_rates = {}
        yesmen = []
        nones = []
        for ticker, container in self._voteData.items():
            latest = container.get_latest()
            if latest is not None:
                if latest.df["No"].isnull().all() or BM:
                    yesmen.append(container.symbol)
                else:
                    approval_rates[container.symbol] = latest.AvgApprovalRate
            else:
                nones.append(container.symbol)

        if len(approval_rates) == 0 and len(yesmen) == 0:
            return

        yesmen_weight = 1 / (len(yesmen) + len(approval_rates))
        yesmen_targets = [PortfolioTarget(yessir, yesmen_weight) for yessir in yesmen]

        targets = [
            PortfolioTarget(symbol, rate / sum(approval_rates.values())) for symbol, rate in approval_rates.items()
        ]
        none_targets = [PortfolioTarget(symbol, 0) for symbol in nones]
        self.SetHoldings(yesmen_targets + targets + none_targets)

    def _checkPortfolio(self):
        values = {}
        for ticker, sec in self.Portfolio.items():
            if sec.Invested:
                values[str(ticker)] = np.round((sec.Quantity * sec.Price) / self.Portfolio.TotalHoldingsValue * 100, 2)
        """self.Debug(
            f"On {self.Time.strftime('%Y-%m-%d')}, Portfolio value: {self.Portfolio.TotalHoldingsValue} with holdings (% of portfolio): {values}"
        )"""
        self._logged_portfolio[self.Time.year] = values

    def OnData(self, data):
        """OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        """
        if self._changes is None:
            return

        for sym in self._symbols:

            if self._symbols[sym] in data:
                try:
                    val = data.Get(SECReport8K, self._symbols[sym])
                except KeyNotFoundException:
                    self.Debug(f"Couldn't get datapoint for {str(sym)}")
                    continue

                if identify(val):
                    if str(sym) in FALSE_POSITIVES and val.get_Time().strftime("%Y-%m-%d") in FALSE_POSITIVES[str(sym)]:
                        continue
                    meeting = MeetingData(decode(val), val.get_Time(), sym, self.Debug)

                    if meeting.df is not None and len(meeting.df.index) > 0:
                        if meeting.extra is not None:
                            self.Debug(
                                f"The following were found for but not selected for {str(meeting.symbol)} at {meeting.date.strftime('%Y-%m-%d')}"
                            )
                            self.Debug(meeting.extra)

                        self._voteData[str(meeting.symbol)].add_year(meeting)

                        # if str(meeting.symbol) == "GOOG":
                        #    self.Debug(meeting.df.to_json(orient="table"))

                        # Saving meeting dates
                        if self.ObjectStore.ContainsKey(str(meeting.symbol)):
                            stored_dates = self.ObjectStore.Read(str(meeting.symbol)).split(",")
                        else:
                            stored_dates = []

                        date = meeting.date.strftime("%Y-%m-%d")
                        if date not in stored_dates:
                            stored_dates.append(date)
                            self.ObjectStore.Save(str(meeting.symbol), ",".join(stored_dates))

                    else:
                        self.Debug(
                            f"Meeting identified no votes: {str(meeting.symbol)} at {meeting.date.strftime('%Y-%m-%d')}"
                        )
                        # self.Debug(f"Reported failures: {[vote.get_text() for vote in meeting.bad]}")
                        # self.Debug(f"Reported extra: {meeting.extra}")
                        # self.Debug(f"Soup: {meeting.meeting}")

        """for security in self._changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)

        for security in self._changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 1 / self._numberOfSymbols)
        """

    def OnSecuritiesChanged(self, changes):
        self._changes = changes

        if self._changes is None:
            return

        for security in self._changes.AddedSecurities:
            self.Log(f"Security: {security} added")
            self._symbols[security.Symbol] = self.AddData(SECReport8K, security.Symbol, Resolution.Daily).Symbol

        self.Log(f"OnSecuritiesChanged({self.UtcTime}):: {changes}")

        # if not self.Portfolio.Invested:
        #    self.SetHoldings("SPY", 1)

    def OnEndOfAlgorithm(self):
        data_keys = [str(j).split(",")[0][1:] for _, j in enumerate(self.ObjectStore.GetEnumerator())]
        saved = {key: self.ObjectStore.Read(key) for key in data_keys}
        self.Debug(f"Saved: {saved}")

        missing_companies = [key for key in self._universe if key not in data_keys]
        self.Debug(f"The following companies are missing: {missing_companies}")
        self.Debug(f"The following weights were observed: {self._logged_portfolio}")


def google_finder(soup):
    prop_key = soup.tr.td.get_text(strip=True).strip("¨")
    subject = soup.tr.find_all("td")[1].get_text(strip=True)

    if len(prop_key) == 0 or len(subject) < 5:
        return

    data = soup.next_sibling.next_sibling
    if type(data) is bs4.element.Tag:
        # See: https://stackoverflow.com/questions/5917082/regular-expression-to-match-numbers-with-or-without-commas-and-decimals-in-text
        base_reg = "(?:^|\s)(?=.)((?:0|(?:[1-9](?:\d*|\d{0,2}(?:,\d{3})*)))?(?:\.\d*[1-9])?)(?!\S)"
        criteria = {
            "For": "votes for",
            "Against": "votes against",
            "Abstain": "abstentions",
            "BrokerNon-Votes": "broker non-votes",
        }
        regs = {key: base_reg + " " + c for key, c in criteria.items()}
        found = {
            key: re.findall(reg, data.get_text())[0]
            for key, reg in regs.items()
            if len(re.findall(reg, data.get_text())) == 1
        }
        df = pd.DataFrame([list(found.values())], columns=list(found.keys()))
        df = df.replace(["", "%"], np.nan).dropna(axis=1, how="all").dropna(axis=0, how="all")
        df["sub_no"] = df.index
        df["explain"] = subject

        if len(df.index) > 0:
            return df


def adobe_select(obj, soup):
    return obj.search_key(soup.parent.parent)


def jpm_select(obj, soup):
    search = re.search("\s*Proposal (\d+)\s?-\s?(.*)", soup.parent.parent.previous_sibling.previous_sibling.get_text())

    if search is not None:
        return search.group(1), search.group(2)

    else:
        search = re.search("\s*Proposal (\d+)\s?-\s?(.*)", soup.parent.parent.previous_sibling.get_text())

        if search is not None:
            return search.group(1), search.group(2)


def amzn_select(obj, soup):
    for sib in soup.parent.parent.previous_siblings:
        if type(sib) is bs4.element.Tag and bool(re.match("\s*ITEM\xa05.07.\s?\s?SUBMISSION", sib.get_text())):
            try:
                return "1", soup.parent.parent.previous_sibling.previous_sibling.get_text()
            except:
                return None


def pg_select(obj, soup):
    directors = re.search("\s*(\d+)\s*.\xa0 (Election of Directors)", soup.div.get_text())
    if directors is not None:
        return directors.group(1), directors.group(2)
    else:
        try:
            old = soup.find_next("tr").find_next("tr").find_next("tr").get_text()
            if "Election of Directors" in old:
                return "Election of Directors", "Election of Directors"
        except:
            pass
    '''else:
        if "Proposal" in soup.div.get_text():
            return "2", "Proposal"'''


class MeetingSettings:
    setting = {
        "default": {
            "yes": ["for", "votesfor"],
            "no": ["against", "votesagainst"],
            "abstain": [
                "voteswithheld",
                "brokernonvotes",
                "brokernonvote",
                "abstain",
                "abstained",
                "withheld",
                "abstentions",
                "authoritywithheld",
            ],
            "nominee": ["person", "directornominee", "nominee", "director", "name"],
        },
        "GOOG": {"yes": ["3years", "2years", "1year"]},
        "AAPL": {"yes": ["1year", "2years", "3years"], "abstain": ["authoritywithheld\nbrokernonvote"]},
        "NFLX": {"yes": ["oneyear", "twoyears", "threeyears"]},
        "PG": {"yes": ["1year", "2years", "3years"], "nominee": ["1.electionofdirectors"], "abstain": ["abstaintions"]},
        "ADBE": {"abstain": ["nonvotes"]},
        "ORCL": {"nominee": ["director’sname", "director\x92sname"]},
    }

    custom_table_finder = {"GOOG": google_finder}
    custom_search = {
        "ADBE": adobe_select,
        "JPM": jpm_select,
        "AMZN": amzn_select,
        "NFLX": adobe_select,
        "MA": adobe_select,
        "PG": pg_select,
    }

    def __init__(self, ticker):
        for attr, val in self.setting["default"].items():
            setattr(self, attr, val)

        if ticker in self.setting:
            for attr, val in self.setting[ticker].items():
                expanded = getattr(self, attr) + val
                setattr(self, attr, expanded)

            # self.yes = self.setting[ticker]["yes"]
            # self.no = self.setting[ticker]["no"]
            # self.abstain = self.setting[ticker]["abstain"]
            # self.nominee = self.setting[ticker]["nominee"]
        if ticker in self.custom_table_finder:
            self.table_constructor = self.custom_table_finder[ticker]

        if ticker in self.custom_search:
            self.table_search = self.custom_search[ticker]

    def find_cols(self, df, selection):
        if df is None or len(df.index) == 0:
            return None

        return df[df.columns.intersection(selection)]

    def sum_yes(self, df):
        return self.find_cols(df, self.yes).sum(axis=1)

    def sum_no(self, df):
        return self.find_cols(df, self.no).sum(axis=1)

    def sum_votes(self, df):
        return self.sum_yes(df) + self.sum_no(df)

    def sum_abstain(self, df):
        return self.find_cols(df, self.abstain).sum(axis=1)


class MeetingData:
    def __init__(self, meeting, date, symbol, debug):
        self.debug = debug
        self.meeting = meeting
        self.symbol = symbol
        self.date = date
        self.select = MeetingSettings(str(self.symbol))
        (
            self.df,
            self.extra,
            self.bad,
        ) = self.process(self.meeting, date)

    def process(self, agm, date):
        process_soup = BeautifulSoup(agm, "lxml")
        # process_soup = BeautifulSoup(agm, "html.parser")
        soup_tables = process_soup.find_all("table")
        tables, fails = self.table_data(soup_tables)
        if len(tables) > 0:
            df = pd.concat(tables, ignore_index=True, sort=False)
            # df["date"] = date

            df.set_index(["vote_no", "explain", "sub_no"], inplace=True)

            num_col_names = self.select.yes + self.select.no + self.select.abstain
            try:
                nums = (
                    df[df.columns.intersection(num_col_names)]
                    .replace({",": "", "—": np.nan, "\x97": np.nan}, regex=True)
                    .apply(pd.to_numeric, 1)
                )
            except ValueError:
                self.debug(f"Had an error interpreting numbers and used fallback option")
                ignored = nums = (
                    df[df.columns.intersection(num_col_names)]
                    .replace({",": "", "—": np.nan, "\x97": np.nan}, regex=True)
                    .apply(pd.to_numeric, 1, errors="ignore")
                )
                nums = ignored[ignored.apply(lambda x: int in list(x.apply(type)), 1)]
                if len(nums.index) == 0:
                    return None, None, fails

            """summary = pd.concat(
                [self.select.sum_yes(df), self.select.sum_no(df), self.select.sum_abstain(df), df[self.select.nominee]],
                axis=1,
            )"""

            summary = pd.concat(
                [
                    self.select.sum_yes(nums),
                    self.select.sum_no(nums),
                    self.select.sum_abstain(nums),
                    df[df.columns.intersection(self.select.nominee)],
                ],
                axis=1,
            )
            # Drop rows with no valid numbers
            summary = summary.drop(summary.index.difference(nums.dropna(axis=0, how="all").index)).dropna(
                axis=1, how="all"
            )
            if len(summary.index) == 0:
                return None, None, fails

            # For identifying data in the wrong place
            trimmed = df.loc[df.index.intersection(summary.index)]
            unexplained_cols = list(
                trimmed.drop(trimmed.columns.intersection(num_col_names + self.select.nominee), axis=1)
                .dropna(axis=1, how="all")
                .columns
            )
            if len(unexplained_cols) > 0:
                excess = unexplained_cols
            else:
                excess = None
            # excess = summary.drop(summary.columns.intersection(num_col_names + self.select.nominee), axis=1)

            col_names = ["Yes", "No", "Abstain", "Nominee"]
            summary.columns = col_names[: len(summary.columns)]

            summary["PassRate"] = np.round((summary["Yes"] / (summary["Yes"] + summary["No"]) * 100), 2)

            summary["AbstainRate"] = np.round(
                summary["Abstain"] / (summary["Yes"] + summary["No"] + summary["Abstain"]) * 100, 2
            )

            return summary, excess, fails
        else:
            return None, None, fails

    def table_data(self, soup_tables):
        tables = []
        fails = []

        for table in soup_tables:
            try:
                prop_key, subject = self.search_key(table)
                if prop_key is None:
                    if hasattr(self.select, "table_search"):
                        prop_key, subject = self.select.table_search(self, table)
                        if prop_key is None:
                            continue
                    else:
                        continue
            except:
                continue

            dfs, fail = self.create_table(table, prop_key, subject)
            if len(dfs) == 0:
                if hasattr(self.select, "table_constructor"):
                    custom_table = self.select.table_constructor(table)
                    if custom_table is not None:
                        tables.append(custom_table)
                fails.append(fail)
            else:
                for df in dfs:
                    tables.append(df)

        votes = []
        for i, vote in enumerate(tables):
            vote["vote_no"] = i
            votes.append(vote)

        return votes, fails

    def search_key(self, soup):
        loop_count = 0
        for sib in soup.previous_siblings:
            if type(sib) is bs4.element.Tag and bool(re.match("\s*\d+", sib.get_text())):
                results = re.findall("\d+", sib.get_text())
                if len(results) > 0:
                    prop_key = results[0]
                    return results[0], sib.get_text()
            elif type(sib) is bs4.element.Tag and bool(
                re.match("Item\xa0\d+.\d+. Submission of Matters to a Vote of Security Holders", sib.get_text())
            ):
                results = re.findall(
                    "Item\xa0\d+.\d+. Submission of Matters to a Vote of Security Holders", sib.get_text()
                )
                if len(results) > 0:
                    prop_key = results[0]
                    return results[0], sib.get_text(strip=True)
            if loop_count > 20:
                break
            loop_count += 1

        return self.backup_key(soup)

    def backup_key(self, soup):
        content = {}
        try:
            content[1] = soup.find_previous("p").get_text()
        except:
            pass

        try:
            content[2] = soup.find_previous("p").find_previous("p").get_text()
        except:
            try:
                content[2] = soup.find_previous("table").get_text()
            except:
                pass

        try:
            content[3] = soup.find_previous("p").find_previous("p").find_previous("p").get_text()
        except:
            pass

        try:
            content[4] = soup.find_previous("p").find_previous("p").find_previous("p").find_previous("p").get_text()
        except:
            pass

        try:
            content["parentVote"] = soup.parent.find_previous_sibling("div").p.get_text()
        except:
            pass

        crit = [
            # ("^Election of Directors", 1),
            ("^\d+", 2),
            ("^\d+", 3),
            ("^Proposal \d+\s*", 3),
            ("\s*ITEM\xa05.07.\s?\s?SUBMISSION", 4),
            ("Item\xa0\d+.\d+. Submission of Matters to a Vote of Security Holders", "parentVote"),
        ]

        for c in crit:
            if c[1] in content and bool(re.match(c[0], content[c[1]])):
                try:
                    key = re.findall("\d+", content[c[1]])[0]
                    return key, content[2]
                except:
                    pass

            if c[1] == 1:
                return content[1], content[1]

            if c[1] == "parentVote":
                try:
                    key = re.findall(
                        "Item\xa0\d+.\d+. Submission of Matters to a Vote of Security Holders", content["parentVote"]
                    )[0]
                    return key, content["parentVote"]
                except:
                    pass

        if 1 in content:
            search = re.search("\s*Proposal\s?(\d+)\s?-\s?(.*)", content[1])
            if search is not None:
                return search.group(1), search.group(2)

        return None, content

    def create_table(self, table, vote_no, explain):
        rows = table.find_all("tr")
        dfs = []
        df = self.interpret_table(rows, explain)

        if df is None:
            msft_division, explainers = self.msft_joined_tables_check(rows)
            if len(msft_division) > 0:
                for i, sub_table in enumerate(msft_division):
                    sub_df = self.interpret_table(sub_table, explainers[i])
                    if sub_df is not None:
                        dfs.append(sub_df)
            else:
                # Failed
                return dfs, table
        else:
            dfs.append(df)
        return dfs, None

    def interpret_table(self, rows, explain):
        row_list = []
        header_row = None
        for i, row in enumerate(rows):
            if type(row) is bs4.element.Tag:
                cols = row.find_all("td")
                col_text = [col.get_text(strip=True) for col in cols]
                if header_row is None:
                    prop_row = [
                        col.lower().replace("\xa0", "").replace("-", "").replace(" ", "")
                        for col in col_text
                        if len(col) > 0
                    ]
                    if len(prop_row) > 0 and len([col for col in prop_row if col.lower() in self.select.yes]) > 0:
                        header_row = prop_row
                else:
                    if "%" in "".join(col_text) and str(self.symbol) == "JPM":
                        continue
                    elif "Independent Auditor" in "".join(col_text) and str(self.symbol) == "MSFT":
                        break
                    row_list.append(col_text)

        if header_row is None:
            return None

        df_full = pd.DataFrame(row_list)
        df = df_full.replace(["", "%", "N/A"], np.nan).dropna(axis=1, how="all").dropna(axis=0, how="all")

        if df is None:
            return None

        if len(header_row) + 2 == len(df.columns):
            df.drop(columns=0, inplace=True)

        if len(header_row) + 1 == len(df.columns):
            header_row.insert(0, "person")

        if len(header_row) == len(df.columns):
            df.columns = header_row
            df["sub_no"] = df.index
            df["explain"] = explain
            return df

    def msft_joined_tables_check(self, rows):
        results = []
        explainers = []
        for i, row in enumerate(rows):
            try:
                if bool(re.match("\xa0\xa0Vote result", row.td.p.get_text())):
                    results.append(i)
            except AttributeError:
                pass

        sub_tables = []
        for i, row_no in enumerate(results):
            explainers.append(rows[row_no - 1].td.get_text())
            if len(results) > i + 1:
                sub_tables.append(rows[row_no : results[i + 1] - 1])
            else:
                sub_tables.append(rows[row_no:])

        return sub_tables, explainers

    @property
    def Passed(self) -> pd.DataFrame:
        return self.df[self.df["PassRate"] >= 50]

    @property
    def Failed(self) -> pd.DataFrame:
        return self.df[self.df["PassRate"] < 50]

    @property
    def PassRate(self):
        if len(self.Passed.index) > 0:
            return np.round(len(self.Passed.index) / (len(self.Passed.index) + len(self.Failed.index)) * 100, 2)
        else:
            return 0

    @property
    def AvgApprovalRate(self):
        return np.round(self.df["PassRate"].mean(), 2)


class MeetingContainer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = str(symbol)
        self._latestdate = None

    def add_year(self, meeting: MeetingData):
        setattr(self, str(meeting.date.year), meeting)
        if self._latestdate is None or self._latestdate < meeting.date:
            self._latestdate = meeting.date

    def get_latest(self):
        if self._latestdate is None:
            return
        else:
            return getattr(self, str(self._latestdate.year))
