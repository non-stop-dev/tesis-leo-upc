## Data Objects


| Data | A data object describing a homogeneous graph. |
| --- | --- |
| HeteroData | A data object describing a heterogeneous graph, holding multiple node and/or edge types in disjunct storage objects. |
| Batch | A data object describing a batch of graphs as one big (disconnected) graph. |
| TemporalData | A data object composed by a stream of events describing a temporal graph. |
| Dataset | Dataset base class for creating graph datasets. |
| InMemoryDataset | Dataset base class for creating graph datasets which easily fit into CPU memory. |
| OnDiskDataset | Dataset base class for creating large graph datasets which do not easily fit into CPU memory at once by leveraging aDatabasebackend for on-disk storage and access of data objects. |


## Remote Backend Interfaces


| FeatureStore | An abstract base class to access features from a remote feature store. |
| --- | --- |
| GraphStore | An abstract base class to access edges from a remote graph store. |
| TensorAttr | Defines the attributes of aFeatureStoretensor. |
| EdgeAttr | Defines the attributes of aGraphStoreedge. |


## Databases


| Database | Base class for inserting and retrieving data from a database. |
| --- | --- |
| SQLiteDatabase | An index-based key/value database based onsqlite3. |
| RocksDatabase | An index-based key/value database based onRocksDB. |


## PyTorch Lightning Wrappers


| LightningDataset | Converts a set ofDatasetobjects into apytorch_lightning.LightningDataModulevariant. |
| --- | --- |
| LightningNodeData | Converts aDataorHeteroDataobject into apytorch_lightning.LightningDataModulevariant. |
| LightningLinkData | Converts aDataorHeteroDataobject into apytorch_lightning.LightningDataModulevariant. |


## Helper Functions


| makedirs | Recursively creates a directory. |
| --- | --- |
| download_url | Downloads the content of an URL to a specific folder. |
| download_google_url | Downloads the content of a Google Drive ID to a specific folder. |
| extract_tar | Extracts a tar archive to a specific folder. |
| extract_zip | Extracts a zip archive to a specific folder. |
| extract_bz2 | Extracts a bz2 archive to a specific folder. |
| extract_gz | Extracts a gz archive to a specific folder. |


