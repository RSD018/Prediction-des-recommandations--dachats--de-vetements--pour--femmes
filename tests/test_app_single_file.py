"""Intégrité de src/app.py : un seul fichier, donc un seul espace de noms partagé par toutes les pages.

Dans un script Streamlit, une variable globale d'une page peut écraser une fonction d'une autre page
(par exemple `cramers_v` est une variable de la page « Tests d'hypothèses »). Ces tests le détectent.
"""
import ast
import builtins

from conftest import BEGIN, END, ROOT

IMPORTS = {"alt", "np", "pd", "st", "Path"}


def _bound_names(nodes):
    """Noms liés au niveau module (y compris dans les if/for/with), hors corps de fonctions."""
    names = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for n in nodes:
        visit(n)
    return names


def _split():
    source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(BEGIN)) + 1
    stop = next(i for i, l in enumerate(lines) if l.startswith(END)) + 1
    tree = ast.parse(source)
    inside = [n for n in tree.body if start <= n.lineno <= stop]
    outside = [n for n in tree.body if not start <= n.lineno <= stop]
    return inside, outside


def test_added_block_names_never_collide_with_the_other_pages():
    inside, outside = _split()
    clash = (_bound_names(inside) & _bound_names(outside)) - IMPORTS
    assert not clash, f"noms définis à la fois dans le bloc ajouté et ailleurs dans app.py : {sorted(clash)}"


def test_added_block_names_are_never_reassigned_by_a_page():
    inside, outside = _split()
    own = _bound_names(inside) - IMPORTS
    reassigned = sorted({n.id for top in outside for n in ast.walk(top)
                         if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in own})
    assert not reassigned, f"une page réaffecte des noms du bloc ajouté : {reassigned}"


def test_added_block_only_defines_things():
    """Le bloc ne doit rien afficher à l'import : seulement des définitions, imports et constantes."""
    inside, _ = _split()
    allowed = (ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)
    offenders = [n.lineno for n in inside if not isinstance(n, allowed)
                 and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert not offenders, f"instructions exécutées au chargement, lignes : {offenders}"


def test_added_block_uses_only_names_that_exist():
    """Un nom introuvable ne se verrait qu'au clic sur la page concernée : on le détecte ici."""
    inside, outside = _split()
    known = _bound_names(inside) | _bound_names(outside) | set(dir(builtins))
    missing = set()
    for top in inside:
        for fn in [n for n in ast.walk(top) if isinstance(n, ast.FunctionDef)]:
            local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    local.add(sub.id)
                elif isinstance(sub, ast.FunctionDef):
                    local.add(sub.name)
                    local.update(a.arg for a in sub.args.args)
                elif isinstance(sub, ast.Lambda):
                    local.update(a.arg for a in sub.args.args)
                elif isinstance(sub, ast.ExceptHandler) and sub.name:
                    local.add(sub.name)
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id not in local | known:
                    missing.add(sub.id)
    assert not missing, f"noms utilisés dans le bloc ajouté mais définis nulle part : {sorted(missing)}"
