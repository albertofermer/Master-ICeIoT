/* global use, db */
// MongoDB Playground
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.

const database = 'EJERCICIO_1';
const collection1 = 'DEPT';
const collection2 = 'EMP';

// Seleccionar (o crear) la base de datos
use(database);

// Crear las colecciones
db.createCollection(collection1);
db.createCollection(collection2);

// Insertar datos en la colección DEPT (Departamentos)
db[collection1].insertMany([
    {
      "DEPTNO": 10,
      "DNAME": "Contabilidad",
      "LOC": "Madrid"
    },
    {
      "DEPTNO": 20,
      "DNAME": "Ventas",
      "LOC": "Barcelona"
    },
    {
      "DEPTNO": 30,
      "DNAME": "Investigación",
      "LOC": "Valencia"
    }
]);

// Insertar datos en la colección EMP (Empleados)
db[collection2].insertMany([
    {
      "EMPNO": 1001,
      "ENAME": "Juan Pérez",
      "JOB": "Analista",
      "HIREDATE": new Date("2023-03-15"),
      "SAL": 2500,
      "COMM": 500,
      "DEPTNO": 10,
      "JEFE": null,
      "SUBORDINADOS": [1002, 1003]
    },
    {
      "EMPNO": 1002,
      "ENAME": "Ana Gómez",
      "JOB": "Desarrolladora",
      "HIREDATE": new Date("2022-06-20"),
      "SAL": 2200,
      "COMM": 300,
      "DEPTNO": 10,
      "JEFE": 1001,
      "SUBORDINADOS": []
    },
    {
      "EMPNO": 1003,
      "ENAME": "Carlos Ruiz",
      "JOB": "Técnico",
      "HIREDATE": new Date("2021-09-10"),
      "SAL": 1800,
      "COMM": null,
      "DEPTNO": 10,
      "JEFE": 1001,
      "SUBORDINADOS": []
    },
    {
      "EMPNO": 2001,
      "ENAME": "Sofía López",
      "JOB": "Vendedora",
      "HIREDATE": new Date("2022-12-01"),
      "SAL": 2000,
      "COMM": 700,
      "DEPTNO": 20,
      "JEFE": null,
      "SUBORDINADOS": []
    }
]);

// Verificar las inserciones
db[collection1].find().pretty();
db[collection2].find().pretty();
