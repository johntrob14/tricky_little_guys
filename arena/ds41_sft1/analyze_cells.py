"""The eval cell names of ds41_sft1, shared by secret_names.py and report.sh."""
CELLS = {"sft": dict(easy="ds41sft1-easy", medium="ds41sft1-medium", control="ds41sft1-ctl-easy", absent="ds41sft1-absent-easy"),
         "rw": dict(easy="ds41rw-cf-easy", medium="ds41rw-cf-medium", control="ds41rw-cf-ctl-easy", absent="ds41rw-cf-absent-easy"),
         "base": dict(easy="ds41base-cf-easy", medium="ds41base-cf-medium")}
ALL_CELLS = [c for m in CELLS.values() for c in m.values()]
