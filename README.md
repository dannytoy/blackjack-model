# Blackjack ML Model vs Basic Strategy & Hi-Lo System

### Public Site

* Main Site Hosted on Vercel Here: https://blackjack-model.vercel.app
* Backend Hosted on Render Here: https://blackjack-model-backend.onrender.com
* Render will spin down with inactivity, so open link if getting error while searching for users

### Prerequisites

*   Python 3.8+
*   Node.js 18+

### Installation & Running on Local

**Backend:**

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the FastAPI server:
    ```bash
    uvicorn main:app --reload --port 8000
    ```
    The API will be available at `http://127.0.0.1:8000`.

**Frontend:**

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install the required Node.js packages:
    ```bash
    npm install
    ```
3.  Run the Next.js development server:
    ```bash
    npm run dev
    ```
    The web application will be available at `http://localhost:3000`.
