# %%
from datetime import datetime
from pprint import pprint
from pathlib import Path

import requests
from typer import Typer
from rich import print as rprint

import config


BASE = "https://api.ynab.com/v1/budgets/last-used/transactions"
HEADERS = {"Authorization": f"Bearer {config.YNAB_TOKEN}"}
SESTERCE_TRANSACTION_URL = "https://app.sesterce.io/groups/txerrkyw/spendings/{id}"

app = Typer()


def is_same(sesterce_row, ynab_transaction):
    # Check less than 3 days appart
    date = sesterce_row["Date"]
    t_date = datetime.strptime(ynab_transaction["date"], "%Y-%m-%d")
    if abs((t_date - date).days) > 3:
        return False

    # Check same amount. YNAB is in milliunits
    amount = sesterce_row["Paid by Diego"]
    if -amount * 1000 != ynab_transaction["amount"]:
        return False

    # If it's a zero transaction, if the subtransactions are the same
    if amount == 0:
        if len(ynab_transaction["subtransactions"]) != 2:
            return False
        for sub in ynab_transaction["subtransactions"]:
            # print(sub, sesterce_row)
            # round to avoid .6666667 problems
            if round(abs(sub["amount"]) - sesterce_row["Paid for Diego"] * 1000) == 0:
                return False

    return True


def match_sesterce_ynab(sesterce_row, ynab_transactions):
    for transaction in ynab_transactions:
        if is_same(sesterce_row, transaction):
            return transaction
    return None


def create_split_0_transaction(
    amount: float, payee_name: str, date: datetime, no_confirm: bool, id: str
):
    payload = dict(
        account_id="d6585d76-0931-4930-bbc2-2e3a4c97393b",  # "Not my transaction" category
        date=date.strftime("%Y-%m-%d"),
        amount=0,
        payee_name=payee_name,
        subtransactions=[
            dict(
                amount=int(-amount * 1000), category_id="7db9d4cf-5c1b-4eba-8f44-6676689287e2"
            ),  # Unknow category
            dict(
                amount=int(amount * 1000), category_id="c586144c-2f67-45b2-a47f-571cabae1d16"
            ),  # Coloc tricount category
        ],
    )

    if not no_confirm:
        print("🤔 Create split transaction in YNAB?")
        rprint(f"   [bold orange3]{payee_name}[/] - {date} - ±{amount}")
        link = SESTERCE_TRANSACTION_URL.format(id=id)
        rprint(f"   🔗 [link={link} u]Sesterce url")

        from InquirerPy import inquirer

        if not inquirer.confirm("Continue?", default=True).execute():
            print("🚫 Skipped")
            return

    response = requests.post(BASE, headers=HEADERS, json={"transaction": payload})

    if response.status_code != 201:
        print(f"❌ Error while creating transaction for {payee_name}: {response.status_code}")
        pprint(response.json())
    else:
        print(f"🌟 Created split 0 transaction for {payee_name}")


def add_split_to_my_transaction(transaction, row, no_confirm):
    paid_for_me = int(row["Paid for Diego"] * 1000)
    paid_by_me = int(row["Paid by Diego"] * 1000)
    # Avoid rounding problems, where parts don't sum up in the sesterce export
    paid_for_coloc = paid_by_me - paid_for_me
    payload = dict(
        id=transaction["id"],
        memo=f"Sesterce: {row['Title']}",
        subtransactions=[
            dict(amount=-paid_for_me, category_id=transaction["category_id"]),
            dict(
                amount=-paid_for_coloc, category_id="c586144c-2f67-45b2-a47f-571cabae1d16"
            ),  # Coloc tricount category
        ],
    )

    if not no_confirm:
        print(f"🤔 Update transaction splits?")
        print(f"    Sesterce: {row['Title']} - {row['Date']}")
        for name, amount in row.items():
            if name.startswith("Paid for"):
                print(f"     {name[9:]: >12}: {amount}€")
        print(
            f"    YNAB: {transaction['payee_name']} - {transaction['date']} ({transaction['amount'] / 1000}€)"
        )
        category = transaction["category_name"] or "[Uncategorized]"
        print(f"      New split: {paid_for_me / 1000}€ {category}")
        print(f"      New split: {paid_for_coloc / 1000}€ Coloc Split")
        from InquirerPy import inquirer

        if not inquirer.confirm("Continue?", default=True).execute():
            print("🚫 Skipped")
            return

    response = requests.put(
        f"{BASE}/{transaction['id']}", headers=HEADERS, json={"transaction": payload}
    )

    if response.status_code != 200:
        print(f"❌ Error while updating transaction for {row['Title']}: {response.status_code}")
        pprint(response.json())
    else:
        print(f"🌟 Updated transaction for {row['Title']} with split")


def sesterce_api_to_df(data: dict):
    """Convert the sesterce API response to a DataFrame with the same columns as their CSV export."""

    import pandas as pd

    spendings = data["spendings"]
    contributors = {d["id"]: d["name"] for d in data["contributors"]}

    rows = []
    for spending in spendings:
        # Example: {'id': '1714828970474_WEB-R6GLQ120ZS67U7MH_SPENDING_bsr4', 'description': 'Migration from Tricount', 'date': '20240504', 'financed': [{'contributor': '1714827951917_WEB-R6GLQ120ZS67U7MH_CONTRIBUTOR_gpsk', 'type': 'fixed', 'value': 36392}], 'spent': [{'contributor': '1714827393333_WEB-R6GLQ120ZS67U7MH_CONTRIBUTOR_v1lc', 'type': 'part', 'value': 100}, {'contributor': '1714827951917_WEB-R6GLQ120ZS67U7MH_CONTRIBUTOR_gpsk', 'type': 'part', 'value': 0}, {'contributor': '1714827388259_WEB-R6GLQ120ZS67U7MH_CONTRIBUTOR_i5d4', 'type': 'fixed', 'value': 8530}, {'contributor': '1714827388259_WEB-R6GLQ120ZS67U7MH_CONTRIBUTOR_i5d4', 'type': 'part', 'value': 0}], 'images_id': [], 'created_at': '2024-05-04T13:22:50+00:00', 'updated_at': '2024-05-04T13:22:50+00:00'}

        row = {
            "Date": spending["date"],
            "Title": spending["description"],
            "ID": spending["id"],
        }
        # Initialize the columns
        for name in contributors.values():
            row[f"Paid for {name}"] = 0
            row[f"Paid by {name}"] = 0

        # First compute+record how much was spent
        total = 0
        for payment in spending["financed"]:
            assert payment["type"] == "fixed"
            row[f"Paid by {contributors[payment['contributor']]}"] += payment["value"] / 100
            total += payment["value"] / 100

        # Then add the fixed parts + track the fractional parts
        parts = 0
        fixed = 0
        for payment in spending["spent"]:
            if payment["type"] == "fixed":
                row[f"Paid for {contributors[payment['contributor']]}"] += payment["value"] / 100
                fixed += payment["value"] / 100
            else:
                assert payment["type"] == "part"
                parts += payment["value"]

        # Split the reminder according to fractional parts
        for payment in spending["spent"]:
            if payment["type"] == "part":
                row[f"Paid for {contributors[payment['contributor']]}"] += (
                    payment["value"] / parts * (total - fixed)
                )

        # Sanity check spent == financed
        spent = sum(row[f"Paid for {name}"] for name in contributors.values())
        financed = sum(row[f"Paid by {name}"] for name in contributors.values())
        assert abs(spent - financed) < 0.01

        rows.append(row)

    return pd.DataFrame(rows)


@app.command()
def sesterce(input_file: Path = None, no_confirm: bool = False, since: str = "last"):
    """Import split transactions from Sesterce to YNAB."""

    # %%

    import pandas as pd
    import utils

    since_file = utils.DATA / "sesterce_last_import"
    if since == "last":
        if since_file.exists():
            since = since_file.read_text().strip()
        else:
            since = "2000-01-01"
    elif since.endswith("d"):
        since = datetime.now() - pd.Timedelta(since)
        since = since.strftime("%Y-%m-%d")
    elif since == "all":
        since = "2000-01-01"

    try:
        since = datetime.strptime(since, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid date format: {since}, should be YYYY-MM-DD, 'last' or 'all'")
        return

    print(f"📅 Importing transactions since {since}")

    # %%

    # Retreive the all transactions from YNAB
    response = requests.get(BASE, headers=HEADERS)
    transactions = response.json()["data"]["transactions"]

    # Filter out old transactions
    transactions = [t for t in transactions if datetime.strptime(t["date"], "%Y-%m-%d") >= since]

    # %% Load the transactions from the CSV file

    # input_file = Path("~/Downloads/export_txerrkyw.csv").expanduser()
    if input_file is None:
        # Read from the sesterce API
        data = requests.get(
            config.SESTERCE_ENDPOINT, headers={"Authorization": config.SESTERCE_AUTH}
        ).json()
        df = sesterce_api_to_df(data)
    else:
        df = pd.read_csv(input_file)
    # print(df.columns)
    # Index(['Date', 'Title', 'Paid by Alexandre', 'Paid by Angélina',
    #    'Paid by Diego', 'Paid for Alexandre', 'Paid for Angélina',
    #    'Paid for Diego', 'Currency', 'Category', 'Exchange rate'],
    #   dtype='object')

    # Make date a datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    # Filter out old transactions
    df = df[df["Date"] >= since]

    # %%

    # For each transaction
    # 0. Try to find the transaction in YNAB
    # 1. If not found, create it
    # 2. If found and already split, do nothing
    # 3. If found and not split, create two splits

    for i, row in df.iterrows():
        if row["Paid by Diego"] == 0 and row["Paid for Diego"] == 0:
            rprint(
                f"🤔 Skipping transaction [bold orange3]{row['Title']}[/] with no amount for Diego"
            )
            continue

        match = match_sesterce_ynab(row, transactions)

        if match and match["subtransactions"]:
            rprint(f"👌 Transaction for [bold orange3]{row['Title']}[/] already present and split")
        elif match:
            add_split_to_my_transaction(match, row, no_confirm)
        elif row["Paid by Diego"] == 0:
            create_split_0_transaction(
                row["Paid for Diego"], row["Title"], row["Date"], no_confirm, row["ID"]
            )
        else:
            link = SESTERCE_TRANSACTION_URL.format(id=row["ID"])
            rprint(
                f"🙈 Did not create entry for my transaction (present [u gray link={link}]in sesterce[/], not ynab) "
                f"on [bold orange3]{row["Title"]}[/] ({row['Date']}), even if missing. [not implemented]"
            )

    # Save the last import date
    since_file.write_text(datetime.now().strftime("%Y-%m-%d"))


@app.command(name="ids")
def ynab_ids():
    """Print the ids of the categories and accounts in YNAB."""
    import requests
    import config

    headers = {"Authorization": f"Bearer {config.YNAB_TOKEN}"}
    # Categories
    response = requests.get(
        "https://api.youneedabudget.com/v1/budgets/last-used/categories", headers=headers
    )

    for group in response.json()["data"]["category_groups"]:
        print(f"📦 {group['name']}")
        for category in group["categories"]:
            print(f"  - {category['name']} ({category['id']})")

    # Accounts
    response = requests.get(
        "https://api.youneedabudget.com/v1/budgets/last-used/accounts", headers=headers
    )
    for account in response.json()["data"]["accounts"]:
        print(f"🏦 {account['name']} ({account['id']})")


@app.command()
def process_bas(input_file: Path, output_file: Path):
    """Convert a bank record from the BAS to the CSV format of YNAB."""
    import warnings

    warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
    import pandas as pd

    df = pd.read_csv(input_file.open(encoding="latin1"), skiprows=11, sep=";", dtype=str)

    assert list(df.columns) == ["Date", "Libellé", "Montant", "Valeur"]

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%y").dt.strftime("%m/%d/%Y")
    out["Payee"] = df["Libellé"]
    out["Memo"] = ""
    out["Amount"] = df["Montant"]

    print(out)
    out.to_csv(output_file, index=False)

    print(f"🎉 Converted {input_file} to {output_file}")
