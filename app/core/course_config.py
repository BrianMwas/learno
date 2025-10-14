"""
Course configuration and curriculum definition.
"""

# Cell Biology Curriculum - Natural progression from outer to inner
CELL_BIOLOGY_CURRICULUM = [
    "Introduction to Cells",
    "Cell Membrane",
    "Cytoplasm",
    "Nucleus",
    "Mitochondria",
    "Endoplasmic Reticulum",
    "Golgi Apparatus",
    "Ribosomes",
    "Lysosomes",
    "Vacuoles",
    "Chloroplasts (Plant Cells)",
    "Cell Wall (Plant Cells)",
    "Cytoskeleton",
    "Cell Types: Prokaryotic vs Eukaryotic",
    "Plant vs Animal Cells"
]

# Map course topics to their curricula
COURSE_CURRICULA = {
    "Cell Biology": CELL_BIOLOGY_CURRICULUM,
}


def get_curriculum(course_name: str) -> list[str]:
    """Get curriculum for a specific course."""
    return COURSE_CURRICULA.get(course_name, CELL_BIOLOGY_CURRICULUM)
