# %%
from datetime import datetime
from pprint import pprint
from pathlib import Path

import requests
from typer import Typer

import config


BASE = "https://api.ynab.com/v1/budgets/last-used/transactions"
HEADERS = {"Authorization": f"Bearer {config.YNAB_TOKEN}"}


app = Typer()


def is_same(sesterce_row, ynab_transaction):
    # Check less than 3 days appart
    date = sesterce_row["Date"]
    t_date = datetime.strptime(ynab_transaction['date'], "%Y-%m-%d")
    if abs((t_date - date).days) > 3:
        return False

    # Check same amount. YNAB is in milliunits
    amount = sesterce_row["Paid by Diego"]
    if -amount * 1000 != ynab_transaction['amount']:
        return False

    # If it's a zero transaction, if the subtransactions are the same
    if amount == 0:
        if len(ynab_transaction['subtransactions']) != 2:
            return False
        for sub in ynab_transaction['subtransactions']:
            if abs(sub['amount']) != sesterce_row["Paid for Diego"] * 1000:
                return False

    return True


def match_sesterce_ynab(sesterce_row, ynab_transactions):
    for transaction in ynab_transactions:
        if is_same(sesterce_row, transaction):
            return transaction
    return None


def create_split_0_transaction(amount: float, payee_name: str, date: datetime, no_confirm: bool):
        payload = dict(
            account_id="d6585d76-0931-4930-bbc2-2e3a4c97393b",  # "Not my transaction" category
            date=date.strftime("%Y-%m-%d"),
            amount=0,
            payee_name=payee_name,
            subtransactions=[
                dict(amount=int(-amount * 1000), category_id="7db9d4cf-5c1b-4eba-8f44-6676689287e2"),  # Unknow category
                dict(amount=int(amount * 1000), category_id="c586144c-2f67-45b2-a47f-571cabae1d16"),  # Coloc tricount category
            ]
        )

        if not no_confirm:
            print(f"🤔 Create split transaction?")
            print(f"    Sesterce: {payee_name} - {date} - ±{amount}")
            if input("Continue? [y/N]: ") != "y":
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
            id=transaction['id'],
            memo=f"Sesterce: {row['Title']}",
            subtransactions=[
                dict(amount=-paid_for_me, category_id=transaction['category_id']),
                dict(amount=-paid_for_coloc, category_id="c586144c-2f67-45b2-a47f-571cabae1d16"),  # Coloc tricount category
            ]
        )

        if not no_confirm:
            print(f"🤔 Update transaction splits?")
            print(f"    Sesterce: {row['Title']} - {row['Date']}")
            for name, amount in row.items():
                if name.startswith("Paid for"):
                    print(f"     {name[9:]: >12}: {amount}€")
            print(f"    YNAB: {transaction['payee_name']} - {transaction['date']} ({transaction['amount'] / 1000}€)")
            category = transaction['category_name'] or "[Uncategorized]"
            print(f"      New split: {paid_for_me / 1000}€ {category}")
            print(f"      New split: {paid_for_coloc / 1000}€ Coloc Split")
            if input("Continue? [y/N]: ") != "y":
                print("🚫 Skipped")
                return

        response = requests.put(f"{BASE}/{transaction['id']}", headers=HEADERS, json={"transaction": payload})

        if response.status_code != 200:
            print(f"❌ Error while updating transaction for {row['Title']}: {response.status_code}")
            pprint(response.json())
        else:
            print(f"🌟 Updated transaction for {row['Title']} with split")


@app.command()
def sesterce(input_file: Path, no_confirm: bool = False, since: str = "last"):
    """Import split transactions from Sesterce to YNAB."""

    import pandas as pd
    import utils

    since_file = utils.DATA / "sesterce_last_import"
    if since == "last":
        if since_file.exists():
            since = since_file.read_text().strip()
        else:
            since = "2000-01-01"
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
    transactions = response.json()['data']['transactions']

    # Filter out old transactions
    transactions = [t for t in transactions if datetime.strptime(t['date'], "%Y-%m-%d") >= since]

    # %% Load the transactions from the CSV file

    # input_file = Path("~/Downloads/export_txerrkyw.csv").expanduser()
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
    df

    #%%

    # For each transaction
    # 0. Try to find the transaction in YNAB
    # 1. If not found, create it
    # 2. If found and already split, do nothing
    # 3. If found and not split, create two splits


    for i, row in df.iterrows():
        match = match_sesterce_ynab(row, transactions)

        if match and match['subtransactions']:
            print(f"👌 Transaction for {row['Title']} already present and split")
        elif match:
            add_split_to_my_transaction(match, row, no_confirm)
        elif row["Paid by Diego"] == 0:
            create_split_0_transaction(row["Paid for Diego"], row["Title"], row["Date"], no_confirm)
        else:
            print(f"🙈 Did not create entry for my transaction on {row["Title"]}, even if missing. [not implemented]")


    # Save the last import date
    since_file.write_text(datetime.now().strftime("%Y-%m-%d"))


@app.command(name="ids")
def ynab_ids():
    """Print the ids of the categories and accounts in YNAB."""
    import requests
    import config

    headers = {"Authorization": f"Bearer {config.YNAB_TOKEN}"}
    # Categories
    response = requests.get("https://api.youneedabudget.com/v1/budgets/last-used/categories", headers=headers)


    for group in response.json()['data']['category_groups']:
        print(f"📦 {group['name']}")
        for category in group['categories']:
            print(f"  - {category['name']} ({category['id']})")

    # Accounts
    response = requests.get("https://api.youneedabudget.com/v1/budgets/last-used/accounts", headers=headers)
    for account in response.json()['data']['accounts']:
        print(f"🏦 {account['name']} ({account['id']})")


@app.command()
def process_bas(input_file: Path, output_file: Path):
    """Convert a bank record from the BAS to the CSV format of YNAB."""
    import warnings
    warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
    import pandas as pd

    df = pd.read_csv(input_file.open(encoding="latin1"),
                     skiprows=11,
                      sep=";", dtype=str)

    assert list(df.columns) == ["Date", "Libellé", "Montant", "Valeur"]

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%y").dt.strftime("%m/%d/%Y")
    out["Payee"] = df["Libellé"]
    out["Memo"] = ""
    out["Amount"] = df["Montant"]

    print(out)
    out.to_csv(output_file, index=False)

    print(f"🎉 Converted {input_file} to {output_file}")
