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
  console.log("===========================");

  document.getElementById("results").classList.remove("hidden");
  document.getElementById("sentencesSection").classList.remove("hidden");

  // Calcul du score global à partir des phrases
  const sentences = data.sentence_analysis || [];

  const humanCount = sentences.filter(
    s => s.label === "HUMAN"
  ).length;

  const aiCount = sentences.filter(
    s => s.label === "AI"
  ).length;

  const totalCount = sentences.length;

  let prediction = "MIXED";
  let score = 50;

  if (humanCount > aiCount) {
    prediction = "HUMAN";
    score = Math.round((humanCount / totalCount) * 100);
  } else if (aiCount > humanCount) {
    prediction = "AI";
    score = Math.round((aiCount / totalCount) * 100);
  }

  document.getElementById("prediction").innerText = prediction;
  document.getElementById("score").innerText = score + "%";

  document.getElementById("language").innerText =
    data.language;

  document.getElementById("lexical").innerText =
    data.lexical_richness;

  document.getElementById("burstiness").innerText =
    data.burstiness;

  document.getElementById("perplexity").innerText =
    data.perplexity;

  document.getElementById("suspicious").innerText =
    aiCount + " / " + totalCount;

  const sentencesDiv = document.getElementById("sentences");

  sentencesDiv.innerHTML = "";

  sentences.forEach(item => {
    const div = document.createElement("div");

    let cssClass = "human";

    if (item.label === "AI") {
      cssClass = "ai";
    } else if (item.label === "MIXED") {
      cssClass = "mixed";
    }

    div.className = "sentence " + cssClass;

    div.innerHTML = `
      <p>${item.sentence}</p>
      <strong>${item.label}</strong> — Score : ${Math.round(item.score * 100)}%
    `;

    sentencesDiv.appendChild(div);
  });
}
