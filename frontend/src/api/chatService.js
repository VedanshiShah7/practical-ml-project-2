import axios from "axios";

export const sendMessage = async (message) => {
  try {
    const response = await axios.post("/api/chat", { message });
    return response.data;
  } catch (error) {
    console.error("Chat service error:", error);
    return { error: "Failed to send message" };
  }
};
