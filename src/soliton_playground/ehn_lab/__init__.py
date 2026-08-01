"""SB-1 lab for the `ehn-two-scalar` preset — box, battery, catalog, field store.

Migrated from `null-worldtube-private/simulations/engine_dogfood/` on 2026-08-01,
that repo being deprecated. The split follows its own EXTRACTION_DECISIONS.md:
the *engine* went to `jax_solitons.ehn` (a library concern, versioned and
installable), and everything here is the *program's* apparatus around it.

  standard_box    SB-1, the frozen reference box, plus the B1-B7 conformance
                  battery and the certificate it issues.
  chamber         the envelope as arithmetic -- preflight walls that answer
                  before any compute is spent (`preflight`, `validate_stage_plan`).
  particle_catalog  registration: a relaxed state is admitted only with measured
                  validity, its topology re-traced rather than trusted.
  field_store     content-addressed storage for the states themselves.
  particles/      the ten catalog entries (metadata; the states are in the store).

WHY field_store EXISTS, since it is the newest piece and the others predate it:
the catalog pinned every field by sha256 and kept it at one gitignored path, so
all ten states were lost while their descriptions survived intact. Verification
therefore meant re-relaxing at N=192, which needs a 24 GB GPU. Three attempts at
the trefoil's determinant failed in transport, never in physics.

The envelope constants here are EHN-gauged and transfer to NOTHING else -- C is a
Chern-Simons coefficient, lam/kappa the EHN potential, el_mag an electric/magnetic
ratio. There is no envelope for the GPE or bare-Faddeev presets, which is why
`docs/CENSUS_PROTOCOL.md` still lists gate -1 as MISSING for those.
"""
