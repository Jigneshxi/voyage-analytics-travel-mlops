# Agent Behavioral Guidelines & Automation Rules

## 📌 Automated Study Guide Saving Rule (Mandatory)
For all agent interactions with the user (Jignesh):
1. **Zero Manual Action Required:** The user must NEVER be asked to run a `.bat`, script, or manual program to save chats.
2. **Automatic Background Archiving:** After completing tasks, explanations, or coding sessions:
   - Automatically compile a structured study guide in Markdown (`.md`).
   - Save it under `Jignesh_Chat_Agent/<YYYY-MM-DD>/<Topic_Name>_Study_Guide.md`.
   - Update `Jignesh_Chat_Agent/README.md` with the new entry in the chronological index.
3. **Structure of Study Guides:**
   - `# 📚 Study Guide: <Topic>`
   - `## 🎯 1. Business Context & Objective`
   - `## 🏗️ 2. Technical Architecture & Formulas`
   - `## 💻 3. Implementation Code & Commands`
   - `## 💡 4. Interview & Evaluation Q&A`
   - `## 📝 5. Key Takeaways & Revision Notes`
