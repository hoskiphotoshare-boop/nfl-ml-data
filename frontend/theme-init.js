/* Early theme bootstrap: runs before first paint. */
(function () {
  try {
    var saved = localStorage.getItem('nfl_theme');
    document.documentElement.dataset.theme = saved === 'dark' ? 'dark' : 'light';
  } catch (e) {
    document.documentElement.dataset.theme = 'light';
  }
})();
