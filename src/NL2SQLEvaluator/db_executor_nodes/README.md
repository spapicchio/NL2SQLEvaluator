## 📖 Overview
This package contains the code for executing queries over the database and possibly use cached data.

## 📑 Documentation Structure
```
├── 📁 db_executor_nodes/                       
│   ├── 📁 cache/                                    # Folder with code related to caching query results
|   │   ├── 📄 cache_protocol.py                     # Protocol defining the interface for caching query results
|   │   ├── 📄 code_normalizer.py                    # Utility for normalizing code to ensure consistent caching
|   │   └── 📄 sqlite_cache.py                       # Implementation of the cache protocol using SQLite
│   ├── 📄 db_executor_protocol.py                   # Protocol defining the interface for executing database queries
│   ├── 📄 sqlite_db_executor.py                     # Implementation of the database executor protocol using SQLite
│   └── 📄 output_table.py                           # Implementaiton of the base classes representing the output of a query execution
```

## 👷🏼‍♂️ How to contribute

If you want to contribute adding a different database executor, you can follow these steps:

1. define in the output table the output of the query execution, following the pydantic class `GenericOutExecutedCode`.
2. create a new python module that implements the `DBExecutorProtocol` defined in `db_executor_protocol.py`
3. create test to cover the new implementation of the database executor protocol. You can find examples of tests in `test_db_executor_sqlite.py`.

Please note that the implementation of `DBExecutorProtocol.execute_queries` must be thread and process safe, 
as it may be called concurrently from multiple threads or processes. 
This means that you should ensure that any shared resources (such as database connections or caches) are properly synchronized to prevent
race conditions and ensure data integrity.

