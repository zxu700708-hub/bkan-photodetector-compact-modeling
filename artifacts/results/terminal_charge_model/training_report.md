# Locally audited terminal-charge compact model

- Production rows: 2080 (160 conditions)
- Strict validation conditions: 16
- Selected alpha: 0.001
- Selected parameter degree: 1
- Selected derivative weight: 10
- Held-out Q test R2: 0.99990153
- Held-out Q active p95 relative error: 1.6686%
- Held-out dQ/dV test p95 relative error: 3.1506%
- Full refit Q active p95 relative error: 1.5580%
- Full refit dQ/dV p95 relative error: 0.7737%
- Q(0 V)=0 is enforced analytically by the voltage basis.
- The exported two-terminal branch uses Qn=-Qp and I_dynamic=ddt(Qp).
- External Spectre compilation and DC/AC/transient execution remain required before any simulator-level verification or sign-off claim.
