import pymongo

def connect(database="Milan_CDR_db", collection="Milan_CDR_c"):
    
    client = pymongo.MongoClient("mongodb://afmhuelva:3NZmlzuSchh9J6k4@localhost:27017/")  # Cambia la URL si usas un servidor remoto o autenticación
    db = client[database]  # Nombre de la base de datos
    collection = db[collection]  # Nombre de la colección dentro de la base de datos
    
    return client, db, collection