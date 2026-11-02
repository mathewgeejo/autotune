"""Scoped accessibility fixes for the pinned Gradio 5.49.1 widgets."""
from html import escape


CSS = '''
.tab-container[aria-hidden="true"] { visibility: hidden !important; }
[data-testid="block-info"] { color: #adb3c8 !important; }
button:focus-visible, input:focus-visible, textarea:focus-visible, [tabindex="0"]:focus-visible {
 outline: 2px solid #c4b5fd !important; outline-offset: 3px;
}
.diagnostic-table table { width: 100%; border-collapse: collapse; color: #e5e6ef; font-size: 14px; }
.diagnostic-table caption { text-align: left; padding: 10px 0; color: #c4c9db; font-weight: 600; }
.diagnostic-table th, .diagnostic-table td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #363b50; white-space: nowrap; }
.diagnostic-table th { background: #1b2030; color: #e5e6ef; }
.diagnostic-table .table-scroll { overflow-x: auto; max-width: 100%; }
'''

# Gradio gives its empty placeholders an accessible name on a generic div.
# Name the existing placeholder as an image; do not hide meaningful controls or
# remove them from the accessibility tree. Only child changes are observed.
JS = '''() => {
 const root = document.querySelector('gradio-app') || document;
 const repair = () => root.querySelectorAll('div.empty[aria-label]:not([role])')
   .forEach(element => element.setAttribute('role', 'img'));
 repair();
 let pending = false;
 new MutationObserver(() => {
   if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; repair(); }); }
 }).observe(root, {childList: true, subtree: true});
}'''


def diagnostic_table(title, headers, rows=()):
    text = lambda value: escape('—' if value is None else str(value))
    heading = ''.join('<th scope="col">'+text(value)+'</th>' for value in headers)
    body = ''.join('<tr>'+''.join('<td>'+text(value)+'</td>' for value in row)+'</tr>' for row in rows)
    if not rows:
        body = f'<tr><td colspan="{len(headers)}">Run a comparison to see measurements.</td></tr>'
    return (f'<div class="table-scroll" role="region" aria-label="{escape(title, quote=True)}" tabindex="0">'
            f'<table><caption>{text(title)}</caption><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>')
