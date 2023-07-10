"""
Constants
"""
IDENTIFIER_COLUMNS = ["contractKey", "score_month"]

LABEL = ["target"]

FEATURES_COLUMNS = [
    "months_on_book",
    "ACC011CRT",
    "ACC100CRT",
    "ACC101REV",
    "ACC104CRT",
    "ACC233CRT",
    "ACC234CRT",
    "ACC309NCT",
    "ACC314CRT",
    "Age",
    "ENQ004OTH",
    "LEG200OTH",
    "NumPTPsL24M",
    "NumRPCsL9M",
    "NumRecsL9M",
    "ACC230CRT",
    "ACC001UCR",
    "ACC209CRT",
    "CON001OTH",
    "CON002OTH",
    "Selected",
    "weight",
]

COLUMNS = IDENTIFIER_COLUMNS + LABEL + FEATURES_COLUMNS
