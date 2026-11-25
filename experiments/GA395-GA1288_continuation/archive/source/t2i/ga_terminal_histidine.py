"""Resolve the exact CHARMM ambiguity among C-terminal histidine patches."""
from openmm import app


class NeutralTerminalHistidine(app.ForceField):
    def _getResidueTemplateMatches(self,res,bondedToAtom,templateSignatures=None,
                                   ignoreExternalBonds=False,ignoreExtraParticles=False):
        attempt=super()._getResidueTemplateMatches
        try:
            return attempt(res,bondedToAtom,templateSignatures,ignoreExternalBonds,ignoreExtraParticles)
        except Exception as exc:
            message=str(exc)
            if (res.name!="HIS" or templateSignatures is None or
                not message.startswith("Multiple non-identical matching templates") or
                "HSE-CTER" not in message or "HSD-CTER" not in message):
                raise
            allowed={signature:[t for t in templates if not t.name.startswith(("HSD-CTER","HSP-CTER"))]
                     for signature,templates in templateSignatures.items()}
            template,match=attempt(res,bondedToAtom,allowed,ignoreExternalBonds,ignoreExtraParticles)
            if match is None or not template.name.startswith("HSE-CTER"):
                raise
            print(f"CHARMM terminal histidine {res.index}: {template.name}",flush=True)
            return [template,match]
