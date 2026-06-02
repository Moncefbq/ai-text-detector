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
    body: JSON.stringify({
      text: text
    })
  });

  const data = await response.json();

  console.log("========== DEBUG ==========");
  console.log("XLMR Label :", data.xlmr_label);
  console.log("XLMR Score :", data.xlmr_score);

  console.log("DeBERTa Label :", data.deberta_label);
  console.log("DeBERTa Score :", data.deberta_score);
  console.log("DeBERTa AI Score :", data.deberta_ai_score);

  console.log("Sentence AI Score :", data.sentence_ai_score);
  console.log("Final AI Score :", data.final_ai_score);
  console.log("Final Label :", data.final_label);
  console.log("===========================");

  document.getElementById("results").classList.remove("hidden");
  document.getElementById("sentencesSection").classList.remove("hidden");

  // ==========================
  // RESULTAT GLOBAL BACKEND
  // ==========================

  document.getElementById("prediction").innerText =
    data.final_label;

  let displayScore = 0;

  if (data.final_label === "AI") {
    displayScore = Math.round(
      data.final_ai_score * 100
    );
  }
  else if (data.final_label === "HUMAN") {
    displayScore = Math.round(
      (1 - data.final_ai_score) * 100
    );
  }
  else {
    displayScore = Math.round(
      data.confidence * 100
    );
  }

  document.getElementById("score").innerText =
    displayScore + "%";

  // ==========================
  // INFOS GENERALES
  // ==========================

  document.getElementById("language").innerText =
    data.language;

  document.getElementById("lexical").innerText =
    data.lexical_richness;

  document.getElementById("burstiness").innerText =
    data.burstiness;

  document.getElementById("perplexity").innerText =
    data.perplexity;

  document.getElementById("suspicious").innerText =
    data.suspicious_sentences_count +
    " / " +
    data.total_sentences;

  // ==========================
  // ANALYSE PHRASE PAR PHRASE
  // ==========================

  const sentencesDiv =
    document.getElementById("sentences");

  sentencesDiv.innerHTML = "";

  const sentences =
    data.sentence_analysis || [];

  sentences.forEach(item => {

    const div = document.createElement("div");

    let cssClass = "human";

    if (item.label === "AI") {
      cssClass = "ai";
    }
    else if (
      item.label === "MIXED" ||
      item.risk_level === "MIXED"
    ) {
      cssClass = "mixed";
    }

    div.className =
      "sentence " + cssClass;

    div.innerHTML = `
      <p>${item.sentence}</p>
      <strong>${item.label}</strong>
      — Score : ${Math.round(item.score * 100)}%
    `;

    sentencesDiv.appendChild(div);
  });

  console.log("Affichage terminé.");
}
