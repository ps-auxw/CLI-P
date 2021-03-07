import database

database.open_db()

with database.env.begin(write=True) as txn:
    txn.drop(database.skip_db)
