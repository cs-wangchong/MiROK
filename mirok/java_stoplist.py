
INVALID_RES = {
    "string", "byte", "char", "int", "double", "float", "object", 
    "data", "obj", "list", "lists", "set", "collection", "array", "map", "maps", "tree", "trees", "node",
    "length", "text", "line", "entry", "entries",
    "java", "javax", "util", "utils", "def", "check", "test", "log", "logger",
    "no", "basic", "public", "local", "global", "tmp", "temp", "random", "parent", "parents",   
}

INVALID_OP1 = {
    "equals", "update", "put", 
    "read", "write", "flush", "undo", "redo", "add", "remove", 
    "store", "save", "load", "restore",
    "mark", "reset", "exists", "mkdirs", 
}

INVALID_OP2 = {
    "equals", "update", "put", 
    "read", "write", "flush", "undo", "redo", "add", "remove", 
    "store", "save", "load", "restore",
    "mark", "reset", "exists", "mkdirs", 
}

# INVALID_OP1 = {
#     "exists", "cancel", "read", "write", "delete", "mkdirs", "on", "un", "get", "rollback",
#     "deallocate", "equals", "can", "free", "entries", "go", "add", "remove", "commit",
#     "quietly", "db" ,"file", "connection", "lockable", "log", "redo", "undo", "execute", "clear", "reset"
# }

# INVALID_OP2 = {
#     "quietly", "db" ,"file", "connection", "lockable", 
#     "init", "play", "draw", "store", "send", "offer", "match", "post", "yield", "query", "build", "notify", "delete",
#     "walk", "put", "exists", "mkdirs", "path", "repaint", "load", "activate", "commit",
#     "on", "un", "rollback", "await", "flush", "latch", "obtain", "update", "fire", "handle", "mark",
#     "pause", "log", "visit", "read", "write", "add", "remove", "redo", "undo", "execute", "clear", "reset"
# }