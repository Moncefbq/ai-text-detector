async function analyzeText() {
  const text = document.getElementById("textInput").value;

  if (!text.trim()) {
    alert("Veuillez entrer un texte.");
    return;
  }

  const response = await fetch("http://127.0.0.1:8000/detect", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: text })
  });

  const data = await response.json();

  document.getElementById("results").classList.remove("hidden");
  document.getElementById("sentencesSection").classList.remove("hidden");

  document.getElementById("prediction").innerText = data.xlmr_label;
  document.getElementById("score").innerText = Math.round(data.final_ai_score * 100) + "%";
  document.getElementById("language").innerText = data.language;
  document.getElementById("lexical").innerText = data.lexical_richness;
  document.getElementById("burstiness").innerText = data.burstiness;
  document.getElementById("perplexity").innerText = data.perplexity;
  document.getElementById("suspicious").innerText =
    data.suspicious_sentences_count + " / " + data.total_sentences;

  const sentencesDiv = document.getElementById("sentences");
  sentencesDiv.innerHTML = "";

  data.sentence_analysis.forEach(item => {
    const div = document.createElement("div");

    let cssClass = "human";
    if (item.risk_level === "AI") cssClass = "ai";
    if (item.risk_level === "MIXED") cssClass = "mixed";

    div.className = "sentence " + cssClass;

    div.innerHTML = `
      <p>${item.sentence}</p>
      <strong>${item.label}</strong> — Score : ${Math.round(item.score * 100)}%
    `;

    sentencesDiv.appendChild(div);
  });
}
