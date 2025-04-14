import Fase2_Aplicaciones_Cientificas_y_empresariales.database_credentials as database_credentials

if __name__ == "__main__":
    # Connections parameters
    client, db, collection = database_credentials.connect(database="CDS16", collection="molecules")
    
    