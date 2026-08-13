# Data provenance

## Source boundary

| Source class | Availability | Repository treatment |
|---|---|---|
| Daily sentiment and OHLCV data for 19 assets | Supplied for a course; affirmative redistribution permission has not been established | External input only; never tracked |
| Assignment and lecture material | Instructor-authored; no repository license | Excluded |
| Instructor notebook | Attributed to the instructor; no code license | Excluded; the package implementation is independent |
| Team notebooks and report | Joint academic work | Excluded |
| Related Springer chapter | Open access under CC BY 4.0; this does not automatically cover underlying data | Linked and attributed, not bundled |
| Synthetic fixture | Generated specifically for this package | Included under the repository license |

The original empirical files and source-derived outputs are not distributed. Users who
independently possess authorised access can validate a compatible local copy through the CLI. Raw
file hashes recorded by a local run establish input identity, not redistribution permission.

The historical ticker `FB` is retained for the 2018–2020 sample. Currencies, indices, and
commodities are outside the confirmatory equity panel because their histories and market structures
differ.

## Schema and adjustment

The loader requires unique, ordered dates; positive OHLC and adjusted-close values; nonnegative
volume and news activity; and the expected sentiment columns. All ten primary equities must share
the same 602-session price calendar.

Eight sentiment columns are validated at the data boundary. The registered Bull/Bear ratio uses
six of them: positive, negative, certainty, uncertainty, financial-up, and financial-down scores.
The fear and financial-hype columns are retained as part of the source schema but do not enter the
registered ratio.

Adjusted open is defined as:

`AdjustedOpen = Open × AdjClose / Close`.

This is a retrospective total-return convention and does not demonstrate that the adjustment
factor was observable in real time. Missing price values fail validation. Missing sentiment remains
explicit, and backward filling is prohibited.
