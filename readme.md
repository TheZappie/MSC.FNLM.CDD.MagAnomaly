### *Informative visualization of magnetic dipole anomalies, using Streamlit*

On **Line simulation/sampling**, drag the dashed line to move it, the curved
arrow to rotate it around its midpoint, or either endpoint to reposition it.
Moving and rotating preserve the line length. Release to update the profiles
and their sampled peak-to-peak values. Each profile is
plotted against distance from its first endpoint. Use **Show second line** to
hide or restore Line 2 without moving its endpoints, and **Reset lines** to
restore the default positions of the visible lines.
Below the profile, each visible line shows the shortest horizontal distance from
the target to the survey segment and the straight-line range at that point,
calculated using the horizontal distance and survey altitude.

## Develop

Content main branch of repo is mirrored to streamlit

Test Locally
```uv run streamlit run main.py```

### Style

Code is formatted with ruff:

```ruff format```

Hosted at https://fugromaganomaly.streamlit.app/