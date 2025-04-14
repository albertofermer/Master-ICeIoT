/* global use, db */
// MongoDB Playground
// To disable this template go to Settings | MongoDB | Use Default Template For Playground.
// Make sure you are connected to enable completions and to be able to run a playground.
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.
// The result of the last command run in a playground is shown on the results panel.
// By default the first 20 documents will be returned with a cursor.
// Use 'console.log()' to print to the debug output.
// For more documentation on playgrounds please refer to
// https://www.mongodb.com/docs/mongodb-vscode/playgrounds/

// Select the database to use.
use('db');

// Agregar objetos a las colecciones
// db.cursos.insertMany([
//     { "_id": 1, "titulo": "Curso de Java", "profesor": 1 },
//     { "_id": 2, "titulo": "Curso de C", "profesor": 1 },
//     { "_id": 3, "titulo": "Curso de PHP", "profesor": 2 },
//     { "_id": 4, "titulo": "Curso de Redes", "profesor": 2 },
//     { "_id": 5, "titulo": "Curso de Machine Learning", "profesor": 2 },
//     { "_id": 6, "titulo": "Curso de PHP", "profesor": 3 }
// ])

// db.profesor.insertMany([
//     { "_id": 1, "name": "Maria" },
//     { "_id": 2, "name": "Juan" },
//     { "_id": 3, "name": "Pedro" }
// ])

// db.videos.insertMany([
//     { "_id": 1, "titulo": "Video Java", "curso": 1 },
//     { "_id": 2, "titulo": "Video C", "curso": 2 },
//     { "_id": 3, "titulo": "Video PHP", "curso": 3 },
//     { "_id": 4, "titulo": "Video Redes", "curso": 4 },
//     { "_id": 5, "titulo": "Video Machine Learning", "curso": 5 },
//     { "_id": 6, "titulo": "Video PHP", "curso": 6 },
//     { "_id": 7, "titulo": "Video JDBC", "curso": 1 },
//     { "_id": 8, "titulo": "Video Regresion", "curso": 5 },
//     { "_id": 9, "titulo": "Video Clasificación", "curso": 5 }
// ])

// Realizar la agregación para obtener todos los videos de Juan
db.profesor.aggregate(
    [
        {
            $match: {
                _id: 2
            }
        },
        {
            $lookup: {
                from: 'cursos',
                localField: '_id',
                foreignField: 'profesor',
                as: 'curso'
            }
        },
        {
            $unwind: '$curso'
        },
        {
            $lookup: {
                from: 'videos',
                localField: 'curso._id',
                foreignField: 'curso',
                as: 'videos'
            }
        }
    ]
).pretty()

