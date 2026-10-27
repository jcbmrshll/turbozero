// The colours the views draw on canvases with. The page's theme (the --sg-* variables
// the frontend sets on this frame, SPEC.md §7.5) where one fits, read when drawing, so a
// new theme takes on the next draw; a game's own board and pieces keep their colours
// whatever the theme.

// a theme colour by name: "fg", "well", "s1", ...
export const theme = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--sg-${name}`).trim();

// `a` mixed into `b`, t from 0 (all b) to 1 (all a)
export const mix = (a, b, t) => `color-mix(in srgb, ${a} ${Math.round(t * 100)}%, ${b})`;

// a slightly sunken area of a panel: under a board, a cell just played
export const sunken = () => mix(theme("edge"), theme("panel"), 0.45);

// the games' own colours
export const BLACK = "#2a1802";
export const WHITE = "#faf2de";
export const OTHELLO_GREEN = "#2f6f52";
export const C4_BLUE = "#1263cc";
export const RED = "#cc2a1f";
export const GOLD = "#b08400";
