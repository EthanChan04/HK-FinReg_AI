"""Constants for the first-pass regulatory graph ontology."""

NODE_TYPES = {
    "Regulator",
    "Document",
    "Clause",
    "Topic",
    "Obligation",
    "Risk",
    "Product",
    "InstitutionType",
    "Chunk",
}

RELATIONS = {
    "issued_by",
    "contains",
    "related_to",
    "applies_to",
    "requires",
    "supported_by",
}
