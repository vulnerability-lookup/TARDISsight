"""
CVE corpora used by the experiments.

``PAPER_CVES`` are the cases studied in arXiv:2604.16038 (used by the Tier-1
evaluation). ``EXTENDED_CVES`` adds a broad set of high-profile, well-sighted
exploited CVEs so the hierarchical pooling experiment can estimate a credible
*population* prior — partial pooling is only meaningful when there are enough
"past" CVEs to borrow strength from.
"""

from __future__ import annotations

PAPER_CVES = [
    "CVE-2025-61932",
    "CVE-2025-59287",
    "CVE-2022-26134",
    "CVE-2024-9164",
    "CVE-2025-54236",
    "CVE-2025-8088",
]

# Notable exploited CVEs with substantial sighting histories (counts probed from
# the live API at corpus-construction time, all >> 80 sightings).
_EXTRA_CVES = [
    "CVE-2021-44228",  # Log4Shell
    "CVE-2021-26855",  # ProxyLogon
    "CVE-2022-30190",  # Follina
    "CVE-2023-34362",  # MOVEit
    "CVE-2023-23397",  # Outlook EoP
    "CVE-2024-3400",   # PAN-OS
    "CVE-2023-44487",  # HTTP/2 Rapid Reset
    "CVE-2024-1709",   # ConnectWise ScreenConnect
    "CVE-2025-0282",   # Ivanti Connect Secure
    "CVE-2024-21413",  # Outlook Moniker
    "CVE-2023-20198",  # Cisco IOS XE
    "CVE-2024-23897",  # Jenkins
    "CVE-2022-1388",   # F5 BIG-IP
    "CVE-2021-34527",  # PrintNightmare
    "CVE-2023-22515",  # Confluence
    "CVE-2024-47575",  # FortiManager
    "CVE-2025-31324",  # SAP NetWeaver
    "CVE-2025-29824",  # Windows CLFS
]

EXTENDED_CVES = PAPER_CVES + _EXTRA_CVES
