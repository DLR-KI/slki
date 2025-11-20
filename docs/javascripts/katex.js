// SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
// SPDX-License-Identifier: CC-BY-NC-4.0
//
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$",  right: "$$",  display: true },
      { left: "$",   right: "$",   display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true }
    ],
  })
})
