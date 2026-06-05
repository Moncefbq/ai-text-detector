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

  console.log("========== DEBUG ==========");
  console.log(data);
  console.log("===========================");

  document.getElementById("results").classList.remove("hidden");
  document.getElementById("modelsSection").classList.remove("hidden");
  document.getElementById("styleSection").classList.remove("hidden");
  document.getElementById("sentencesSection").classList.remove("hidden");

  const prediction = document.getElementById("prediction");
  const score = document.getElementById("score");
  const globalBox = document.getElementById("globalBox");

  prediction.innerText = data.final_label || "---";

  let displayScore = 0;

  if (data.final_label === "AI") {
    displayScore = Math.round((data.final_ai_score || 0) * 100);
    globalBox.className = "global-box global-ai";
  } else if (data.final_label === "HUMAN") {
    displayScore = Math.round((1 - (data.final_ai_score || 0)) * 100);
    globalBox.className = "global-box global-human";
  } else {
    displayScore = Math.round((data.confidence || 0) * 100);
    globalBox.className = "global-box global-mixed";
  }

  score.innerText = displayScore + "%";

  document.getElementById("interpretation").innerText =
    "Confiance : " + Math.round((data.confidence || 0) * 100) + "%";

  document.getElementById("language").innerText = data.language || "---";
  document.getElementById("confidence").innerText =
    Math.round((data.confidence || 0) * 100) + "%";

  document.getElementById("suspicious").innerText =
    data.suspicious_sentences_count ?? "---";

  document.getElementById("totalSentences").innerText =
    data.total_sentences ?? "---";

  const xlmrScore = Math.round((data.xlmr_ai_score || 0) * 100);
  const xlmrLargeScore = Math.round((data.xlmr_large_ai_score || 0) * 100);
  const debertaScore = Math.round((data.deberta_ai_score || 0) * 100);
  const sentenceScore = Math.round((data.sentence_ai_score || 0) * 100);
  const styleScore = Math.round((data.stylometry_ai_score || 0) * 100);

  document.getElementById("xlmrBar").style.width = xlmrScore + "%";
  document.getElementById("xlmrLargeBar").style.width = xlmrLargeScore + "%";
  document.getElementById("debertaBar").style.width = debertaScore + "%";
  document.getElementById("sentenceBar").style.width = sentenceScore + "%";
  document.getElementById("styleBar").style.width = styleScore + "%";

  document.getElementById("xlmrValue").innerText = xlmrScore + "%";
  document.getElementById("xlmrLargeValue").innerText = xlmrLargeScore + "%";
  document.getElementById("debertaValue").innerText = debertaScore + "%";
  document.getElementById("sentenceValue").innerText = sentenceScore + "%";
  document.getElementById("styleValue").innerText = styleScore + "%";

  document.getElementById("xlmrLabel").innerText = data.xlmr_label || "---";
  document.getElementById("xlmrLargeLabel").innerText =
    data.xlmr_large_label || "---";
  document.getElementById("debertaLabel").innerText =
    data.deberta_label || "---";

  const s = data.stylometry || {};

  document.getElementById("wordCount").innerText = s.word_count ?? "---";
  document.getElementById("sentenceCount").innerText = s.sentence_count ?? "---";
  document.getElementById("lexical").innerText = s.lexical_richness ?? "---";
  document.getElementById("avgSentence").innerText =
    s.average_sentence_length ?? "---";
  document.getElementById("avgWord").innerText = s.average_word_length ?? "---";
  document.getElementById("repetition").innerText = s.repetition_rate ?? "---";
  document.getElementById("punctuation").innerText =
    s.punctuation_density ?? "---";
  document.getElementById("entropy").innerText = s.entropy ?? "---";

  document.getElementById("burstiness").innerText = data.burstiness ?? "---";
  document.getElementById("perplexity").innerText = data.perplexity ?? "---";

  const sentencesDiv = document.getElementById("sentences");
  sentencesDiv.innerHTML = "";

  const sentences = data.sentence_analysis || [];

  sentences.forEach(item => {
    const div = document.createElement("div");

    let cssClass = "human";

    if (item.label === "AI") {
      cssClass = "ai";
    } else if (item.label === "MIXED" || item.risk_level === "MIXED") {
      cssClass = "mixed";
    }

    div.className = "sentence " + cssClass;

    div.innerHTML = `
      <p>${item.sentence}</p>
      <strong>${item.label}</strong>
      <br>
      Score : ${Math.round((item.score || 0) * 100)}%
      <br>
      Risk : ${item.risk_level}
    `;

    sentencesDiv.appendChild(div);
  });

  console.log("Dashboard avancé avec XLM-R Large chargé.");
}
