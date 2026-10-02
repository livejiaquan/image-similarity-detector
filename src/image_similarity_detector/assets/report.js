(() => {
  const cards = [...document.querySelectorAll(".match-card")];
  const buttons = [...document.querySelectorAll("[data-filter]")];
  const search = document.getElementById("search");
  let mode = "all";
  const apply = () => {
    const term = search.value.toLocaleLowerCase().trim();
    let visible = 0;
    cards.forEach(card => {
      const typeMatch = mode === "all" || card.dataset.kind === mode ||
        (mode === "cross" && card.dataset.cross === "true");
      card.hidden = !(typeMatch && card.textContent.toLocaleLowerCase().includes(term));
      if (!card.hidden) visible++;
    });
    document.getElementById("filter-empty").hidden = visible > 0 || cards.length === 0;
    document.getElementById("visible-count").textContent =
      cards.length ? visible + " of " + cards.length + " displayed cards match your filters." : "";
  };
  buttons.forEach(button => button.addEventListener("click", () => {
    mode = button.dataset.filter;
    buttons.forEach(item => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    apply();
  }));
  search.addEventListener("input", apply);
  apply();
})();

