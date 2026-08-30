"""Licence metadata and composition rules for the source datasets.

Licensing is an engineering constraint here, not paperwork. Two facts drive
everything in this module:

**ShareAlike propagates.** Natural Questions is CC BY-SA 3.0 and HotpotQA is
CC BY-SA 4.0. Derivative works — including the ablated question sets this
pipeline generates — inherit the obligation. Any released artifact built from
them must carry a compatible licence.

**NonCommercial does not compose with ShareAlike.** A single combined artifact
mixing a CC BY-NC source with a CC BY-SA source cannot satisfy both. Such
sources have to stay in a separately licensed component, or stay out.

The safe distribution strategy, which `manifest.py` records: **do not
redistribute the corpora**. Ship loaders, checksums and our own derived
annotations, and let each user obtain the raw data from its original source.

Nothing here is legal advice. It encodes the terms as recorded during dataset
selection so the pipeline can refuse an unsafe combination loudly instead of
producing a quietly unpublishable artifact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LicenseInfo:
    """The terms of one source dataset, as recorded during selection."""

    spdx: str
    name: str
    url: str = ""
    #: Derivatives must be released under the same terms.
    share_alike: bool = False
    #: Commercial use prohibited.
    non_commercial: bool = False
    #: Attribution required.
    attribution: bool = True
    #: Whether we may redistribute the raw corpus, as opposed to a loader.
    redistribution_allowed: bool = True
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


#: Licences of the datasets selected in the corpus-selection review. Verified
#: against primary sources at selection time; re-check before publication, as
#: dataset terms can change between releases.
LICENSES: dict[str, LicenseInfo] = {
    "CC-BY-SA-3.0": LicenseInfo(
        spdx="CC-BY-SA-3.0",
        name="Creative Commons Attribution-ShareAlike 3.0",
        url="https://creativecommons.org/licenses/by-sa/3.0/",
        share_alike=True,
        notes="Natural Questions. Derivatives inherit ShareAlike.",
    ),
    "CC-BY-SA-4.0": LicenseInfo(
        spdx="CC-BY-SA-4.0",
        name="Creative Commons Attribution-ShareAlike 4.0",
        url="https://creativecommons.org/licenses/by-sa/4.0/",
        share_alike=True,
        notes="HotpotQA and SQuAD 2.0. Derivatives inherit ShareAlike.",
    ),
    "CC-BY-4.0": LicenseInfo(
        spdx="CC-BY-4.0",
        name="Creative Commons Attribution 4.0",
        url="https://creativecommons.org/licenses/by/4.0/",
        notes="QASPER and CLAPnq. Most permissive of the selected sources.",
    ),
    "CC-BY-NC-4.0": LicenseInfo(
        spdx="CC-BY-NC-4.0",
        name="Creative Commons Attribution-NonCommercial 4.0",
        url="https://creativecommons.org/licenses/by-nc/4.0/",
        non_commercial=True,
        notes="CRAG. Does not compose with ShareAlike sources in one artifact.",
    ),
    "Apache-2.0": LicenseInfo(
        spdx="Apache-2.0",
        name="Apache License 2.0",
        url="https://www.apache.org/licenses/LICENSE-2.0",
        notes="TriviaQA. Not selected, recorded for completeness.",
    ),
    "UNKNOWN": LicenseInfo(
        spdx="UNKNOWN",
        name="Unverified licence",
        redistribution_allowed=False,
        notes="Terms not verified. Treat as non-redistributable until checked.",
    ),
}


def get_license(spdx: str) -> LicenseInfo:
    """Look up a licence, falling back to the conservative UNKNOWN entry."""
    return LICENSES.get(spdx, LICENSES["UNKNOWN"])


def check_composition(spdx_ids: list[str]) -> list[str]:
    """Problems with combining these licences into one released artifact.

    Empty list means the combination is safe to release as a single unit under
    the strictest of the licences involved.
    """
    problems: list[str] = []
    licenses = [get_license(s) for s in dict.fromkeys(spdx_ids)]

    unknown = [lic.spdx for lic in licenses if lic.spdx == "UNKNOWN"]
    if unknown:
        problems.append(
            "one or more sources have unverified licence terms; verify before release"
        )

    share_alike = [lic for lic in licenses if lic.share_alike]
    non_commercial = [lic for lic in licenses if lic.non_commercial]

    if share_alike and non_commercial:
        problems.append(
            f"ShareAlike ({', '.join(lic.spdx for lic in share_alike)}) cannot be combined "
            f"with NonCommercial ({', '.join(lic.spdx for lic in non_commercial)}) in a "
            "single artifact; keep them as separately licensed components"
        )

    share_alike_versions = {lic.spdx for lic in share_alike}
    if len(share_alike_versions) > 1:
        problems.append(
            f"multiple ShareAlike versions present ({', '.join(sorted(share_alike_versions))}); "
            "confirm one-way compatibility before releasing a combined derivative"
        )
    return problems


def effective_obligations(spdx_ids: list[str]) -> dict:
    """The obligations a derivative of these sources inherits."""
    licenses = [get_license(s) for s in dict.fromkeys(spdx_ids)]
    return {
        "licenses": sorted({lic.spdx for lic in licenses}),
        "attribution_required": any(lic.attribution for lic in licenses),
        "share_alike_required": any(lic.share_alike for lic in licenses),
        "non_commercial_restricted": any(lic.non_commercial for lic in licenses),
        "corpus_redistribution_allowed": all(lic.redistribution_allowed for lic in licenses),
        "composition_problems": check_composition(spdx_ids),
        "recommended_distribution": (
            "Do not redistribute raw corpora. Ship loaders, checksums and derived "
            "annotations keyed by question id; each user obtains raw data from the "
            "original source."
        ),
    }
