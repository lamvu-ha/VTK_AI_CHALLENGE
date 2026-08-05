"""
Milvus collection schema definition.
"""
from typing import Dict, Any


MILVUS_SCHEMA = {
    "collection_name": "vtk_keyframes",
    "fields": [
        {"name": "id",              "dtype": "INT64",       "is_primary": True, "auto_id": True},
        {"name": "video_id",        "dtype": "VARCHAR",     "max_length": 64},
        {"name": "frame_id",        "dtype": "INT64"},
        {"name": "model_name",      "dtype": "VARCHAR",     "max_length": 32},
        {"name": "embedding_vector","dtype": "FLOAT_VECTOR","dim": 512},
    ],
    "index_params": {
        "field_name": "embedding_vector",
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 1024},
    },
}


def get_schema_definition() -> Dict[str, Any]:
    return MILVUS_SCHEMA
