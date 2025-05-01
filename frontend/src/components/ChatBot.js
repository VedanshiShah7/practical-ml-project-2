import React, { useState } from "react";
import axios from "../api/chatService";

function Chatbot({ patient }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const handleSend = async () => {
    if (!input.trim()) return;
    const response = await axios.post("/ask", {
      question: input,
      patient_data: patient,
    });
    setMessages([...messages, { text: input, sender: "user" }, { text: response.data.response, sender: "bot" }]);
    setInput("");
  };

  return (
    <div className="chatbot-container">
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
      </div>
      <input value={input} onChange={(e) => setInput(e.target.value)} />
      <button onClick={handleSend}>Send</button>
    </div>
  );
}

export default Chatbot;