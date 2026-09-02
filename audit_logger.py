"""
audit_logger.py
Tamper-Evident Explainable Audit Logging for LedgerLens AI.

Implements cryptographic SHA-256 hash chaining to provide a verifiable,
tamper-evident audit trail for every transaction lifecycle event across all 7 phases:
  1. OBSERVE      - Ingestion, validation, and normalization
  2. MATCH        - Stage 1 exact and Stage 2 rule evaluation
  3. INVESTIGATE  - AI root-cause analysis for ambiguous leftovers
  4. VERIFY       - Programmatic evidence & mathematical validation
  5. DECIDE       - Confidence policy & risk assessment (AUTO_RESOLVE / HUMAN_REVIEW / ABSTAIN / UNRESOLVED)
  6. ACT          - Execution of resolution or escalation
  7. AUDIT        - Cryptographic recording of all rationale, timestamps, and hashes
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


class AuditLogger:
    """
    Manages structured, explainable, and tamper-evident audit logs using SHA-256 hash chaining.
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        self.start_time = datetime.now()
        self.logs: List[Dict[str, Any]] = []
        self.record_timelines: Dict[str, List[Dict[str, Any]]] = {}
        self.last_hash: str = "0" * 64  # Genesis hash

    def _compute_hash(self, entry_payload: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for a log entry incorporating the previous block's hash."""
        serialized = json.dumps(entry_payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def log_lifecycle_step(
        self,
        record_id: str,
        lifecycle_phase: str,  # OBSERVE | MATCH | INVESTIGATE | VERIFY | DECIDE | ACT | AUDIT
        stage_name: str,       # e.g., "Stage 1 (Exact)", "Stage 2 (Rules)", "Stage 3 (AI)", "Evidence Verifier"
        action: str,
        status: str,           # SUCCESS | REJECTED | ESCALATED | ABSTAINED | ERROR | IN_PROGRESS
        details: Dict[str, Any],
        candidate_ids: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        evidence_summary: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Record a granular, tamper-evident lifecycle event for a transaction.
        """
        timestamp = datetime.now().isoformat()
        
        # Raw entry data before hashing
        entry_data = {
            "run_id": self.run_id,
            "timestamp": timestamp,
            "record_id": record_id,
            "lifecycle_phase": lifecycle_phase.upper(),
            "stage_name": stage_name,
            "action": action,
            "status": status,
            "confidence": confidence,
            "candidate_ids": candidate_ids or [],
            "evidence_summary": evidence_summary or [],
            "details": details,
            "prev_hash": self.last_hash
        }
        
        # Compute entry hash and update chain
        entry_hash = self._compute_hash(entry_data)
        entry_data["entry_hash"] = entry_hash
        self.last_hash = entry_hash
        
        self.logs.append(entry_data)
        self.record_timelines.setdefault(record_id, []).append(entry_data)
        return entry_data

    def verify_audit_integrity(self) -> Tuple[bool, int, Optional[str]]:
        """
        Cryptographically verify the entire SHA-256 audit chain.
        
        Returns:
            Tuple[bool, int, Optional[str]]: (is_valid, verified_count, error_msg_if_any)
        """
        expected_prev_hash = "0" * 64
        for idx, entry in enumerate(self.logs):
            recorded_hash = entry.get("entry_hash")
            recorded_prev = entry.get("prev_hash")
            
            if recorded_prev != expected_prev_hash:
                return False, idx, f"Hash chain broken at index {idx}: expected prev {expected_prev_hash[:8]}..., got {recorded_prev[:8]}..."
                
            # Recompute hash without entry_hash field
            payload = {k: v for k, v in entry.items() if k != "entry_hash"}
            computed = self._compute_hash(payload)
            if computed != recorded_hash:
                return False, idx, f"Data tampering detected at index {idx}: computed {computed[:8]}... != recorded {recorded_hash[:8]}..."
                
            expected_prev_hash = recorded_hash
            
        return True, len(self.logs), None

    def get_timeline_for_record(self, record_id: str) -> List[Dict[str, Any]]:
        """Retrieve the sequential agentic timeline for a given transaction ID."""
        return self.record_timelines.get(record_id, [])

    def export_json(self, indent: int = 2) -> str:
        """Export all logs as a formatted JSON string."""
        return json.dumps(self.logs, indent=indent, default=str)

    def export_summary(self) -> Dict[str, Any]:
        """Summary of the audit run."""
        is_valid, count, err = self.verify_audit_integrity()
        return {
            "run_id": self.run_id,
            "started_at": self.start_time.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "total_events_logged": len(self.logs),
            "distinct_records_tracked": len(self.record_timelines),
            "tamper_evident_integrity": "VERIFIED" if is_valid else "CORRUPTED",
            "cryptographic_chain_head": self.last_hash
        }
