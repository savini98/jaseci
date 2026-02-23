"""MongoDB PersistenceBackend Implementation."""
import logging;
import from collections.abc { Callable, Generator, Iterable }
import from pickle { dumps, loads }
import from typing { Any, cast }
import from uuid { UUID }
import from pymongo { MongoClient, UpdateOne }
import from pymongo.errors { ConnectionFailure }
import from jaclang.jac0core.archetype { Anchor, NodeAnchor, Root }
import from jaclang.runtimelib.utils { storage_key, to_uuid }

glob logger = logging.getLogger(__name__);

"""Serialize archetype attributes to a flat dict of JSON-safe scalars.

Mirrors the same helper in runtime.jac so the adjacency collection only
ever contains values that MongoDB can store and query with equality filters.
Non-scalar values are coerced to str; un-stringifiable values are omitted.
"""
def _safe_attrs(archetype: Any) -> dict[(str, Any)] {
    result: dict = {};
    for (k, v) in archetype.__dict__.items() {
        if k.startswith('_') {
            continue;
        }
        if isinstance(v, (str, int, float, bool)) or v is None {
            result[k] = v;
        } else {
            try {
                result[k] = str(v);
            } except Exception {
                pass;
            }
        }
    }
    return result;
}

"""Post-initialization: connect to MongoDB."""
impl MongoBackend.postinit -> None {
    if self.client is None and self.mongo_url {
        try {
            self.client = MongoClient(self.mongo_url);
            self.db = self.client[self.db_name];
            self.collection = self.db[self.collection_name];
            self.adj_collection = self.db['adjacency'];
            self.adj_collection.create_index([("src", 1), ("tgt_type", 1)]);
            self.adj_collection.create_index([("tgt", 1), ("src_type", 1)]);
            self.adj_collection.create_index("edge_id");
        } except Exception as e {
            logger.debug(f"MongoDB connection failed: {e}");
            self.client = None;
        }
    }
}

"""Check if MongoDB is available and connected."""
impl MongoBackend.is_available -> bool {
    if not self.mongo_url {
        return False;
    }
    client = None;
    try {
        client = MongoClient(self.mongo_url);
        client.admin.command('ping', maxTimeMS=100);
        return True;
    } except ConnectionFailure {
        return False;
    } except Exception as e {
        logger.debug(f"MongoDB availability check failed: {e}");
        return False;
    } finally {
        if client {
            try {
                client.close();
            } except Exception { }
        }
    }
}

"""Get anchor by UUID from MongoDB."""
impl MongoBackend.get(id: UUID) -> (Anchor | None) {
    if self.client is None {
        return None;
    }
    _id = to_uuid(id);
    try {
        db_obj = self.collection.find_one({'_id': str(_id)});
        if db_obj {
            return self._load_anchor(db_obj);
        }
    } except Exception as e {
        logger.debug(f"MongoDB get failed: {e}");
    }
    return None;
}

"""Store anchor in MongoDB and refresh adjacency attrs if it is a NodeAnchor."""
impl MongoBackend.put(anchor: Anchor) -> None {
    if self.client is None or not anchor.persistent {
        return;
    }
    _id = to_uuid(anchor.id);
    try {
        data_blob = dumps(anchor);
        self.collection.update_one(
            {'_id': str(_id)},
            {'$set': {'data': data_blob, 'type': type(anchor).__name__}},
            upsert=True
        );
    } except Exception as e {
        logger.debug(f"MongoDB put failed: {e}");
    }
    if isinstance(anchor, NodeAnchor) and anchor.is_populated() {
        try {
            attrs = _safe_attrs(anchor.archetype);
            self.adj_refresh_node(str(_id), attrs);
        } except Exception as e {
            logger.debug(f"MongoDB put adj_refresh failed: {e}");
        }
    }
}

"""Delete anchor from MongoDB."""
impl MongoBackend.delete(id: UUID) -> None {
    if self.client is None {
        return;
    }
    _id = to_uuid(id);
    try {
        self.collection.delete_one({'_id': str(_id)});
    } except Exception as e {
        logger.debug(f"MongoDB delete failed: {e}");
    }
}

"""Close MongoDB connection."""
impl MongoBackend.close -> None {
    if self.client {
        try {
            self.client.close();
        } except Exception as e {
            logger.warning(f"Error closing MongoDB connection: {e}");
        }
        self.client = None;
    }
}

"""Check if an anchor exists in MongoDB."""
impl MongoBackend.`has(id: UUID) -> bool {
    if self.client is None {
        return False;
    }
    _id = to_uuid(id);
    try {
        return self.collection.count_documents({'_id': str(_id)}, limit=1) > 0;
    } except Exception {
        return False;
    }
}

"""Query all anchors with optional filter."""
impl MongoBackend.query(
    filter: (Callable[[Anchor], bool] | None) = None
) -> Generator[Anchor, None, None] {
    if self.client is None {
        return;
    }
    try {
        for doc in self.collection.find() {
            if (anchor := self._load_anchor(doc)) {
                if filter is None or filter(anchor) {
                    yield anchor;
                }
            }
        }
    } except Exception as e {
        logger.debug(f"MongoDB query failed: {e}");
    }
}

"""Get all root anchors."""
impl MongoBackend.get_roots -> Generator[Root, None, None] {
    for anchor in self.query() {
        if isinstance(anchor.archetype, Root) {
            yield cast(Root, anchor.archetype);
        }
    }
}

"""Find anchors by IDs with optional filter."""
impl MongoBackend.find(
    ids: (UUID | Iterable[UUID]), filter: (Callable[[Anchor], Anchor] | None) = None
) -> Generator[Anchor, None, None] {
    id_list = [ids] if isinstance(ids, UUID) else list(ids);
    for id in id_list {
        if (anchor := self.get(id)) {
            if filter is None or filter(anchor) {
                yield anchor;
            }
        }
    }
}

"""Find one anchor by ID(s) with optional filter."""
impl MongoBackend.find_one(
    ids: (UUID | Iterable[UUID]), filter: (Callable[[Anchor], Anchor] | None) = None
) -> (Anchor | None) {
    id_list = [ids] if isinstance(ids, UUID) else list(ids);
    for id in id_list {
        if (anchor := self.get(id)) {
            if filter is None or filter(anchor) {
                return anchor;
            }
        }
    }
    return None;
}

"""Commit - no-op for MongoDB (writes are immediate)."""
impl MongoBackend.commit(anchor: (Anchor | None) = None) -> None {
# No-op: MongoDB writes are immediate
}

# PersistentMemory-specific methods
"""Sync - no-op for MongoDB (writes are immediate)."""
impl MongoBackend.sync -> None {
# No-op: MongoDB writes are immediate
}

"""Bulk store multiple anchors."""
impl MongoBackend.bulk_put(anchors: Iterable[Anchor]) -> None {
    if self.client is None {
        return;
    }
    ops: list = [];
    for anchor in anchors {
        if not anchor.persistent {
            continue;
        }
        _id = to_uuid(anchor.id);
        try {
            data_blob = dumps(anchor);
            ops.append(
                UpdateOne(
                    {'_id': str(_id)},
                    {'$set': {'data': data_blob, 'type': type(anchor).__name__}},
                    upsert=True
                )
            );
        } except Exception as e {
            logger.debug(f"MongoDB bulk_put serialization failed: {e}");
        }
    }
    if ops {
        try {
            self.collection.bulk_write(ops);
        } except Exception as e {
            logger.debug(f"MongoDB bulk_write failed: {e}");
        }
    }
}

"""Load anchor from raw MongoDB document."""
impl MongoBackend._load_anchor(raw: dict[(str, Any)]) -> (Anchor | None) {
    if 'data' not in raw {
        return None;
    }
    try {
        data: bytes = raw['data'];
        return loads(data);
    } except Exception as e {
        logger.debug(f"MongoDB _load_anchor failed: {e}");
        return None;
    }
}

"""Upsert one entry in the adjacency matrix collection.

Each document represents a single directed edge with enough metadata
to answer type+attribute filter queries without loading the full anchor.
Document _id is src:tgt:edge so duplicate connects are idempotent.
"""
impl MongoBackend.adj_upsert(
    src_id: str,
    tgt_id: str,
    edge_id: str,
    edge_type: str,
    src_type: str,
    tgt_type: str,
    src_attrs: dict[(str, Any)],
    tgt_attrs: dict[(str, Any)]
) -> None {
    if self.client is None {
        return;
    }
    try {
        doc = {
            "src": src_id,
            "tgt": tgt_id,
            "edge_id": edge_id,
            "edge_type": edge_type,
            "src_type": src_type,
            "tgt_type": tgt_type,
            "src_attrs": src_attrs,
            "tgt_attrs": tgt_attrs
        };
        self.adj_collection.update_one(
            {"_id": f"{src_id}:{tgt_id}:{edge_id}"},
            {"$set": doc},
            upsert=True
        );
    } except Exception as e {
        logger.debug(f"adj_upsert failed: {e}");
    }
}

"""Remove all adjacency entries for a given edge (both directions)."""
impl MongoBackend.adj_delete_by_edge(edge_id: str) -> None {
    if self.client is None {
        return;
    }
    try {
        self.adj_collection.delete_many({"edge_id": edge_id});
    } except Exception as e {
        logger.debug(f"adj_delete_by_edge failed: {e}");
    }
}

"""Query the adjacency matrix for connected node IDs without loading archetypes.

Returns a list of UUID strings for nodes that pass the type and attribute
filters, using only the lightweight adjacency index — no anchor deserialization.

direction "OUT": edges leaving origin  (filter applies to target nodes)
direction "IN":  edges entering origin (filter applies to source nodes)
direction "ANY": both directions combined
"""
impl MongoBackend.adj_query(
    origin_id: str,
    node_type: (str | None) = None,
    attrs: (dict[(str, Any)] | None) = None,
    direction: str = "OUT"
) -> list[str] {
    if self.client is None {
        return [];
    }
    try {
        result_ids: list[str] = [];
        if direction in ["OUT", "ANY"] {
            query: dict = {"src": origin_id};
            if node_type {
                query["tgt_type"] = node_type;
            }
            if attrs {
                for (k, v) in attrs.items() {
                    query[f"tgt_attrs.{k}"] = v;
                }
            }
            for doc in self.adj_collection.find(query, {"tgt": 1}) {
                result_ids.append(doc["tgt"]);
            }
        }
        if direction in ["IN", "ANY"] {
            query = {"tgt": origin_id};
            if node_type {
                query["src_type"] = node_type;
            }
            if attrs {
                for (k, v) in attrs.items() {
                    query[f"src_attrs.{k}"] = v;
                }
            }
            for doc in self.adj_collection.find(query, {"src": 1}) {
                if doc["src"] not in result_ids {
                    result_ids.append(doc["src"]);
                }
            }
        }
        return result_ids;
    } except Exception as e {
        logger.debug(f"adj_query failed: {e}");
        return [];
    }
}

"""Refresh adjacency entries when a node's attributes change.

Called automatically from put() whenever a NodeAnchor is persisted.
Updates src_attrs for all edges where this node is the source, and
tgt_attrs for all edges where this node is the target, so subsequent
attribute-filter queries always see the current values.
"""
impl MongoBackend.adj_refresh_node(node_id: str, attrs: dict[(str, Any)]) -> None {
    if self.client is None {
        return;
    }
    try {
        self.adj_collection.update_many(
            {"src": node_id},
            {"$set": {"src_attrs": attrs}}
        );
        self.adj_collection.update_many(
            {"tgt": node_id},
            {"$set": {"tgt_attrs": attrs}}
        );
    } except Exception as e {
        logger.debug(f"adj_refresh_node failed: {e}");
    }
}
