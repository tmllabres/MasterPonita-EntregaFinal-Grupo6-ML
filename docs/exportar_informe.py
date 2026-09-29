"""Exporta el README a docs/informe_final.pdf, que es lo que se sube a PontIA.

    python docs/exportar_informe.py

Convierte el README a HTML con el módulo markdown y lo imprime a PDF con Chrome o Edge
en modo headless (no hace falta pandoc ni LaTeX). El HTML se escribe un momento en la
raíz del repositorio para que las rutas de las figuras (outputs/...) funcionen igual que
en GitHub, y se borra al terminar.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

RAIZ = Path(__file__).resolve().parent.parent
PDF = RAIZ / "docs" / "informe_final.pdf"
REPO = "https://github.com/tmllabres/MasterPonita-ML-EntregaFinal-Grupo6"

NAVEGADORES = [
    Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe",
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
]

ESTILO = """
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       max-width: 860px; margin: 0 auto; color: #111; }
h1 { font-size: 20pt; } h2 { font-size: 14pt; border-bottom: 1px solid #ccc; margin-top: 26px; }
table { border-collapse: collapse; font-size: 8.5pt; margin: 8px 0; }
th, td { border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; }
th { background: #f2f2f2; }
pre { background: #f6f6f6; padding: 8px; font-size: 8pt; white-space: pre-wrap; }
code { font-size: 90%; }
img { max-width: 78%; display: block; margin: 10px auto; page-break-inside: avoid; }
blockquote { color: #444; border-left: 3px solid #ccc; margin-left: 0; padding-left: 12px; }
"""


def navegador() -> str:
    for ruta in NAVEGADORES:
        if ruta.exists():
            return str(ruta)
    for nombre in ("chrome", "msedge", "google-chrome", "chromium"):
        if shutil.which(nombre):
            return shutil.which(nombre)
    sys.exit("No encuentro Chrome ni Edge: abre el README en GitHub e imprímelo a PDF.")


def main() -> None:
    # El markdown dentro de <details> solo se convierte si el bloque lo pide, y en el
    # PDF tiene que verse abierto: impreso, un <details> cerrado no enseña nada.
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    texto = texto.replace("<details>", '<details open markdown="1">')
    cuerpo = markdown.markdown(texto, extensions=["tables", "fenced_code", "md_in_html"])
    # Los enlaces relativos (docs/..., notebooks/...) apuntarían al disco de quien lo
    # exporta: en el PDF van a su página en GitHub. Las imágenes sí se leen del disco.
    cuerpo = re.sub(r'href="(?!https?:|#|mailto:)([^"]+)"',
                    lambda m: f'href="{REPO}/blob/main/{m.group(1)}"', cuerpo)
    html = RAIZ / "_informe_final.html"
    html.write_text(f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
                    f"<title>Informe final</title><style>{ESTILO}</style></head>"
                    f"<body>{cuerpo}</body></html>", encoding="utf-8")
    try:
        subprocess.run([navegador(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", html.as_uri()],
                       check=True, capture_output=True, timeout=120)
    finally:
        html.unlink(missing_ok=True)
    print(f"PDF escrito en {PDF.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
