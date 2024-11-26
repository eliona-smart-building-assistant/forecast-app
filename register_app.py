from config import apps_api


def Initialize():

    app = apps_api.get_app_by_name("forecast")

    if not app.registered:
        apps_api.patch_app_by_name("forecast", True)
        print("App 'forecast' registered.")

    # else:
    #     print("App 'forecast' already active.")
