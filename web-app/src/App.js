import './App.css';
import React from 'react';

function App() {
  const [readingScore, setReadingScore] = React.useState("");
  const [writingScore, setWritingScore] = React.useState("");
  const [selectedGender, setSelectedGender] = React.useState("");
  const [selectedParentEdu, setSelectedParentEdu] = React.useState("");
  const [selectedLunch, setSelectedLunch] = React.useState("");
  const [selectedTestPrep, setSelectedTestPrep] = React.useState("");
  const [response, setResponse] = React.useState(null);

  const optionsGender = ["Kadın", "Erkek"];
  const optionsParentEdu = ["Lise (Mezun Değil)", "Lise Mezunu", "Önlisans", "Lisans (Mezun Değil)", "Lisans", "Yüksek Lisans"];
  const optionsLunch = ["Ücretsiz/İndirimli", "Standart"];
  const optionsTestPrep = ["Tamamlandı", "Yok"];

  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = {
      gender: selectedGender,
      parentEdu: selectedParentEdu,
      lunch: selectedLunch,
      testPrep: selectedTestPrep,
      readingScore: parseInt(readingScore),
      writingScore: parseInt(writingScore),
    }
    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setResponse(Math.round(data.prediction));
      console.log("Response:", data);
      setSelectedGender("");
      setSelectedParentEdu("");
      setSelectedLunch("");
      setSelectedTestPrep("");
      setReadingScore("");
      setWritingScore("");
    } catch (error) {
      console.error("Error:", error);
    }
  }

  return (
    <div className="container">
      <h1>Öğrenci Skor Tahmini Uygulaması</h1>
      <form onSubmit={handleSubmit}>

        <div className="form-group">
          <label>
            Cinsiyet*
          </label>
          <select value={selectedGender} onChange={(e) => setSelectedGender(e.target.value)} required>
            <option value="">Seçiniz</option>
            {optionsGender.map((option, index) => (
              <option key={index} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div className="form-group">      <label>
          Ebeveyn Eğitimi*
        </label>
          <select value={selectedParentEdu} onChange={(e) => setSelectedParentEdu(e.target.value)} required>
            <option value="">Seçiniz</option>
            {optionsParentEdu.map((option, index) => (
              <option key={index} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>
            Öğle Yemeği*
          </label>
          <select value={selectedLunch} onChange={(e) => setSelectedLunch(e.target.value)} required>
            <option value="">Seçiniz</option>
            {optionsLunch.map((option, index) => (
              <option key={index} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>
            Hazırlık Testi*
          </label>
          <select value={selectedTestPrep} onChange={(e) => setSelectedTestPrep(e.target.value)} required>
            <option value="">Seçiniz</option>
            {optionsTestPrep.map((option, index) => (
              <option key={index} value={option}>{option}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Okuma Notu*</label>
          <input type="number" value={readingScore} onChange={(e) => setReadingScore(e.target.value)} min={0} max={100} required />
        </div>
        <div className="form-group">
          <label>Yazma Notu*</label>
          <input type="number" value={writingScore} onChange={(e) => setWritingScore(e.target.value)} min={0} max={100} required />
        </div>
        <button type="submit">Hesapla</button>
      </form >
      {response && (
        <div className="result">
          <h3>Tahmin Sonucu</h3>
          <p>Öğrencinin matematik notu: {response}{" "}
            {response >= 60 ? "✔" : "✖"}</p>
        </div>
      )}
    </div >
  );
}

export default App;
