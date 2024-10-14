# %%
import os

import ovh
import typer

app = typer.Typer()


def get_client() -> ovh.Client:
    application_key = os.getenv("OVH_APPLICATION_KEY")
    application_secret = os.getenv("OVH_APPLICATION_SECRET")
    consumer_key = os.getenv("OVH_CONSUMER_KEY")

    if application_key and application_secret and consumer_key:
        return ovh.Client(
            endpoint="ovh-eu",  # Endpoint of API OVH (List of available endpoints:
            application_key=application_key,  # Application Key
            application_secret=application_secret,  # Application Secret
            consumer_key=consumer_key,  # Consumer Key
        )
    else:
        print("Please set the following environment variables:")
        print("OVH_APPLICATION_KEY")
        print("OVH_APPLICATION_SECRET")
        print("OVH_CONSUMER_KEY")
        print(
            "You can generate new credentials with full access to your account on the token creation page (https://api.ovh.com/createToken/index.cgi?GET=/*&PUT=/*&POST=/*&DELETE=/*)"
        )

        raise typer.Abort()


@app.command(name="list")
def list_domains():
    """
    List all domains
    """

    client = get_client()

    domains: list[str] = client.get("/domain/")
    for domain in domains:
        print(domain)


@app.command(name="get")
def get_zone(domain: str):
    """
    Get zone file for a domain
    """

    client = get_client()
    zone = client.get(f"/domain/zone/{domain}/export")
    print(zone)


@app.command(name="set")
def set_zone(domain: str, zone_file: str):
    """
    Set zone file for a domain
    """
    client = get_client()

    print(f"Setting zone file for {domain}")
    print("---")
    print(zone_file)
    print("---")

    if not typer.confirm("Do you want to continue?"):
        raise typer.Abort()

    result = client.post(
        f"/domain/zone/{domain}/import",
        zoneFile=zone_file,
    )
    print(result)
