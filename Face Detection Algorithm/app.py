from pathlib import Path
from datetime import datetime, timedelta
import csv

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


class FaceAttendanceApp:
    """OOP Streamlit app using OpenCV-only face features for matching."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.images_path = self.base_dir / "Images"
        self.attendance_file = self.base_dir / "Attendance.csv"
        self.rejected_file = self.base_dir / "Attendance_Rejected.csv"
        self.images_path.mkdir(exist_ok=True)

        # Prevent spamming attendance within a short window (in-memory only)
        self.attendance_tracker: dict[str, datetime] = {}

        # Simple classification / metadata defaults
        self.system_version = "1.1"
        # Cosine similarity threshold for OpenCV-based face vectors (0–1)
        self.similarity_threshold = 0.93
        self.class_start_hour = 9
        self.class_start_minute = 0
        self.present_grace_minutes = 15  # <= 15 mins late -> Present; <=60 -> Late; else Rejected

        # Use OpenCV Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # In-memory embeddings (simple grayscale vectors for known faces)
        self.known_names: list[str] = []
        self.known_vectors: list[np.ndarray] = []

        self.load_known_faces()

    # ---------- Utility helpers ----------
    @staticmethod
    def decode_image(uploaded_file) -> np.ndarray:
        """Convert Streamlit camera input to an RGB numpy array."""
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)

    def _face_vector(self, bgr_image: np.ndarray) -> np.ndarray | None:
        """Compute a simple embedding: grayscale face region resized to 64x64 and flattened.

        Returns None if no face is detected.
        """
        try:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                return None

            x, y, w, h = faces[0]
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, (64, 64))
            vec = face_resized.flatten().astype("float32")
            vec /= (np.linalg.norm(vec) + 1e-8)
            return vec
        except Exception:
            # Any failure means we simply couldn't compute a usable face vector
            return None

    def load_known_faces(self) -> None:
        """Load stored face images and compute simple vectors for recognition."""
        self.known_names = []
        self.known_vectors = []

        for image_path in self.images_path.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                vec = self._face_vector(img)
                if vec is None:
                    st.warning(f"No face found in {image_path.name}; skipping.")
                    continue
                self.known_names.append(image_path.stem)
                self.known_vectors.append(vec)
            except Exception as exc:
                st.warning(f"Could not process {image_path.name}: {exc}")

    def _log_reject(
        self,
        name: str | None,
        date_str: str,
        time_str: str,
        day_str: str,
        reason: str,
        confidence: float | None,
        source: str,
    ) -> None:
        """Log rejected or failed attempts to a separate CSV."""
        self.rejected_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.rejected_file.exists()
        write_header = (not file_exists) or self.rejected_file.stat().st_size == 0

        with self.rejected_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "Name",
                        "Date",
                        "Time",
                        "Day",
                        "Reason",
                        "Confidence",
                        "Source",
                        "CreatedAt",
                        "SystemVersion",
                        "SimilarityThreshold",
                    ]
                )
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(
                [
                    name or "",
                    date_str,
                    time_str,
                    day_str,
                    reason,
                    f"{confidence:.4f}" if confidence is not None else "",
                    source,
                    created_at,
                    self.system_version,
                    self.similarity_threshold,
                ]
            )

    def mark_attendance(self, name: str, confidence: float, source: str = "Camera") -> tuple[str, bool]:
        """Append attendance with structured schema and classification.

        Returns (status, recorded) where status is Present/Late/Rejected* and recorded
        indicates whether the row was written to the main attendance CSV.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        day_str = now.strftime("%A")

        # Short-term cooldown (in-memory only)
        last_seen = self.attendance_tracker.get(name)
        if last_seen and (now - last_seen) < timedelta(minutes=30):
            self._log_reject(
                name,
                date_str,
                time_str,
                day_str,
                reason="Cooldown active (recent attendance already recorded)",
                confidence=confidence,
                source=source,
            )
            return "Rejected - cooldown", False

        # One attendance per person per day (check persisted CSV)
        if self.attendance_file.exists():
            try:
                df_existing = pd.read_csv(self.attendance_file, comment="#")
                if "Date" in df_existing.columns and "Name" in df_existing.columns and "Status" in df_existing.columns:
                    mask = (df_existing["Name"].str.lower() == name.lower()) & (df_existing["Date"] == date_str)
                    mask = mask & df_existing["Status"].isin(["Present", "Late"])
                    if mask.any():
                        self._log_reject(
                            name,
                            date_str,
                            time_str,
                            day_str,
                            reason="Duplicate attendance for this date",
                            confidence=confidence,
                            source=source,
                        )
                        return "Rejected - duplicate", False
            except Exception:
                # On read failure, fall back to just in-memory cooldown
                pass

        # Time-based status classification
        class_start = now.replace(
            hour=self.class_start_hour, minute=self.class_start_minute, second=0, microsecond=0
        )
        delta_minutes = (now - class_start).total_seconds() / 60.0
        if delta_minutes <= self.present_grace_minutes:
            status = "Present"
        elif delta_minutes <= 60:
            status = "Late"
        else:
            status = "Rejected - too late"
            self._log_reject(
                name,
                date_str,
                time_str,
                day_str,
                reason="Arrived too late for session",
                confidence=confidence,
                source=source,
            )
            return status, False

        # Persist accepted attendance
        self.attendance_tracker[name] = now
        self.attendance_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.attendance_file.exists()
        write_header = (not file_exists) or self.attendance_file.stat().st_size == 0

        with self.attendance_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "Name",
                        "Date",
                        "Time",
                        "Day",
                        "Status",
                        "Course",
                        "Session",
                        "Confidence",
                        "Source",
                        "CreatedAt",
                        "SystemVersion",
                        "SimilarityThreshold",
                    ]
                )

            # Session / course metadata from sidebar (if set)
            course = st.session_state.get("course", "")
            session = st.session_state.get("session", "")
            created_at = now.strftime("%Y-%m-%d %H:%M:%S")

            writer.writerow(
                [
                    name,
                    date_str,
                    time_str,
                    day_str,
                    status,
                    course,
                    session,
                    f"{confidence:.4f}",
                    source,
                    created_at,
                    self.system_version,
                    self.similarity_threshold,
                ]
            )

        return status, True

    # ---------- Page renderers ----------
    def render_home(self) -> None:
        with st.container():
            st.markdown('<div class="big-title">Face Attendance System</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="subtitle">Minimal, camera-based attendance using Streamlit and OpenCV.</div>',
                unsafe_allow_html=True,
            )
            with st.container():
                st.markdown("### 👣 Quick steps")
                st.markdown(
                    "- **Register**: capture a clear photo with your name.\n"
                    "- **Mark Attendance**: capture again to log a timestamp.\n"
                    "- **View Attendance**: review and download previous records."
                )

    def render_register(self) -> None:
        st.markdown('<div class="big-title">👤 Register</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Capture a clear, front-facing photo to register your face.</div>',
            unsafe_allow_html=True,
        )

        # Keep multiple registration samples in session for more robust matching
        if "reg_samples" not in st.session_state:
            st.session_state["reg_samples"] = []
        if "last_reg_image_bgr" not in st.session_state:
            st.session_state["last_reg_image_bgr"] = None

        name_input = st.text_input("Name", placeholder="Type your full name")
        register_capture = st.camera_input("Camera")

        col_add, col_clear, col_save = st.columns(3)

        with col_add:
            if st.button("Add sample"):
                if register_capture is None:
                    st.error("Capture a photo before adding a sample.")
                else:
                    try:
                        image_rgb = self.decode_image(register_capture)
                        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    except Exception as exc:
                        st.error(f"Could not process camera image: {exc}")
                    else:
                        vec = self._face_vector(image_bgr)
                        if vec is None:
                            st.error("No face detected in this sample. Please try again.")
                        else:
                            st.session_state["reg_samples"].append(vec)
                            st.session_state["last_reg_image_bgr"] = image_bgr
                            st.info(f"Samples captured: {len(st.session_state['reg_samples'])}")

        with col_clear:
            if st.button("Clear samples"):
                st.session_state["reg_samples"] = []
                st.session_state["last_reg_image_bgr"] = None
                st.info("Cleared stored samples.")

        with col_save:
            if st.button("➕ Save face"):
                name = (name_input or "").strip()
                if not name:
                    st.error("Please provide a valid name (not empty).")
                    return
                if any(existing.lower() == name.lower() for existing in self.known_names):
                    st.error("This name is already registered. Please use a different name or update the existing one.")
                    return

                samples: list[np.ndarray] = st.session_state.get("reg_samples", [])
                if len(samples) < 3:
                    st.error("Please add at least 3 good samples before saving.")
                    return

                # Average all collected face vectors to get a more robust representation
                stacked = np.stack(samples, axis=0)
                avg_vec = stacked.mean(axis=0).astype("float32")
                avg_vec /= (np.linalg.norm(avg_vec) + 1e-8)

                image_bgr = st.session_state.get("last_reg_image_bgr")
                if image_bgr is None:
                    if register_capture is None:
                        st.error("No camera image available to save. Capture a photo again and add as sample.")
                        return
                    try:
                        image_rgb = self.decode_image(register_capture)
                        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    except Exception as exc:
                        st.error(f"Could not process camera image for saving: {exc}")
                        return

                save_path = self.images_path / f"{name}.jpg"
                cv2.imwrite(str(save_path), image_bgr)

                # Update in-memory database
                self.known_names.append(name)
                self.known_vectors.append(avg_vec)

                # Reset samples after successful registration
                st.session_state["reg_samples"] = []
                st.session_state["last_reg_image_bgr"] = None

                st.success(f"Registration successful for {name}! (samples used: {len(samples)})")

    def _match_face(self, bgr_image: np.ndarray) -> tuple[str, float] | None:
        """Match captured face against known vectors using cosine similarity.

        Returns (name, confidence) if a strong enough match is found,
        otherwise None. Confidence is the cosine similarity (0–1).
        """
        if not self.known_vectors:
            return None

        vec = self._face_vector(bgr_image)
        if vec is None:
            return None

        sims = [float(np.dot(vec, known_vec)) for known_vec in self.known_vectors]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= self.similarity_threshold:
            return self.known_names[best_idx], best_sim
        return None

    def render_attendance(self) -> None:
        st.markdown('<div class="big-title">✅ Mark Attendance</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Stand in front of the camera and capture a frame to record attendance.</div>',
            unsafe_allow_html=True,
        )
        if not self.known_names:
            st.info("No registered faces yet. Please register first.")
            return

        attendance_capture = st.camera_input("Take a photo to mark attendance")
        if not attendance_capture:
            return

        try:
            image_rgb = self.decode_image(attendance_capture)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            st.error(f"Could not process camera image: {exc}")
            return

        # Quick visual feedback: draw rectangle if face detected
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        match = self._match_face(image_bgr)

        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Captured frame")

        if match:
            matched_name, confidence = match
            status, recorded = self.mark_attendance(matched_name, confidence, source="Camera")
            if recorded:
                st.success(f"Attendance marked for {matched_name} ({status}, confidence={confidence:.3f}).")
            else:
                st.info(f"Attendance not recorded for {matched_name} ({status}).")
        else:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            day_str = now.strftime("%A")
            # Log as rejected attempt with no recognized name
            self._log_reject(
                name=None,
                date_str=date_str,
                time_str=time_str,
                day_str=day_str,
                reason="No reliable face match",
                confidence=None,
                source="Camera",
            )
            st.error(
                "No reliable match found. Make sure the person is registered and the face is clearly visible."
            )

    def render_history(self) -> None:
        """Show previous attendance records using pandas + numpy."""
        st.markdown('<div class="big-title">📊 Attendance History</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Browse and export all previously marked attendance entries.</div>',
            unsafe_allow_html=True,
        )

        if not self.attendance_file.exists():
            st.info("No attendance has been recorded yet.")
            return

        # Load CSV into a DataFrame (handle both legacy and new schemas).
        # Use engine='python' and skip bad lines in case older rows have a different number of columns.
        try:
            df_raw = pd.read_csv(
                self.attendance_file,
                comment="#",
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:
            st.error(f"Could not read attendance file: {exc}")
            return

        if df_raw.empty:
            st.info("Attendance file is empty.")
            return

        # Legacy format: Name,Timestamp -> convert to new-style columns
        if list(df_raw.columns) == ["Name", "Timestamp"] or df_raw.shape[1] == 2:
            df = df_raw.copy()
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"])
            df["Date"] = df["Timestamp"].dt.date.astype(str)
            df["Time"] = df["Timestamp"].dt.time.astype(str)
            df["Day"] = df["Timestamp"].dt.day_name()
            df["Status"] = "Present"
            df["Course"] = ""
            df["Session"] = ""
            df["Confidence"] = np.nan
            df["Source"] = "Camera"
            df["CreatedAt"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df["SystemVersion"] = ""
            df["SimilarityThreshold"] = np.nan
        else:
            df = df_raw.copy()
            if "Time" in df.columns:
                # Keep time as string
                df["Time"] = df["Time"].astype(str)

        # Ensure Date column is proper datetime for any schema
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])

        # Optional filtering by name, date and session
        all_names = sorted(df["Name"].dropna().unique().tolist())
        selected_name = st.selectbox("Filter by name (optional)", options=["All"] + all_names)

        if "Date" in df.columns:
            unique_dates = sorted({str(d.date()) for d in df["Date"].dropna()})
        else:
            unique_dates = []
        selected_date = st.selectbox("Filter by date (optional)", options=["All"] + unique_dates)

        # Session filter: always offer Morning/Evening if the column exists
        if "Session" in df.columns:
            existing_sessions = set(df["Session"].dropna().astype(str).unique().tolist())
            base_sessions = ["Morning", "Evening"]
            all_sessions = [s for s in base_sessions if s in existing_sessions] or base_sessions
            selected_session = st.selectbox(
                "Filter by session (optional)", options=["All"] + all_sessions
            )
        else:
            selected_session = st.selectbox("Filter by session (optional)", options=["All"])

        df_display = df
        if selected_name != "All":
            df_display = df_display[df_display["Name"] == selected_name]
        if selected_date != "All" and "Date" in df_display.columns:
            df_display = df_display[df_display["Date"].dt.date.astype(str) == selected_date]
        if selected_session != "All" and "Session" in df_display.columns:
            df_display = df_display[df_display["Session"] == selected_session]

        # Use numpy to compute simple stats: total records and unique people
        name_array = df_display["Name"].to_numpy()
        total_records = int(name_array.size)
        unique_people = int(np.unique(name_array).size)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total records (filtered)", total_records)
        col2.metric("Unique people (filtered)", unique_people)
        if "Date" in df_display.columns:
            total_days = int(df_display["Date"].dt.date.nunique())
            col3.metric("Distinct days (filtered)", total_days)

        # Hide low-level metadata columns in the main table view
        hide_cols = {"Confidence", "SystemVersion", "SimilarityThreshold", "CreatedAt"}
        cols_to_show = [c for c in df_display.columns if c not in hide_cols]

        st.dataframe(
            df_display[cols_to_show].sort_values(
                ["Date", "Time"] if "Date" in df_display.columns and "Time" in df_display.columns else cols_to_show,
                ascending=False,
                ignore_index=True,
            ),
            use_container_width=True,
        )

        # Per-person attendance percentage (within filtered set)
        if not df_display.empty:
            counts = df_display.groupby("Name").size().rename("Count")
            total = counts.sum()
            perc = (counts / total * 100.0).round(1).rename("Percent")
            summary_df = pd.concat([counts, perc], axis=1).reset_index()
            st.markdown("#### Attendance summary (filtered)")
            st.dataframe(summary_df, use_container_width=True)

        # Weekly / monthly summaries as downloadable CSVs
        if "Date" in df.columns and not df.empty:
            df_dates = df.copy()
            df_dates["YearWeek"] = df_dates["Date"].dt.strftime("%Y-W%U")
            df_dates["YearMonth"] = df_dates["Date"].dt.strftime("%Y-%m")

            weekly = (
                df_dates.groupby(["Name", "YearWeek"])
                .size()
                .rename("Count")
                .reset_index()
                .sort_values(["YearWeek", "Name"])
            )
            monthly = (
                df_dates.groupby(["Name", "YearMonth"])
                .size()
                .rename("Count")
                .reset_index()
                .sort_values(["YearMonth", "Name"])
            )

            st.markdown("#### Export summaries")
            w_csv = weekly.to_csv(index=False)
            m_csv = monthly.to_csv(index=False)
            col_w, col_m = st.columns(2)
            col_w.download_button(
                "Download weekly summary CSV",
                data=w_csv,
                file_name="attendance_weekly_summary.csv",
                mime="text/csv",
            )
            col_m.download_button(
                "Download monthly summary CSV",
                data=m_csv,
                file_name="attendance_monthly_summary.csv",
                mime="text/csv",
            )

        # Download filtered base data as CSV
        csv_content = df_display.to_csv(index=False)
        st.download_button(
            "Download filtered records as CSV",
            data=csv_content,
            file_name="attendance_history.csv",
            mime="text/csv",
        )

        # Show recent rejection log entries (audit trail)
        if self.rejected_file.exists():
            try:
                rej_df = pd.read_csv(self.rejected_file, comment="#")
                if not rej_df.empty:
                    st.markdown("#### Recent rejected attempts")
                    st.dataframe(
                        rej_df.sort_values("CreatedAt", ascending=False).head(20),
                        use_container_width=True,
                    )
            except Exception:
                pass

    # ---------- App runner ----------
    def run(self) -> None:
        st.set_page_config(page_title="Face Attendance (Streamlit)", page_icon="🧠", layout="centered")

        # Global minimal styling
        st.markdown(
            """
            <style>
            body {
                background: radial-gradient(circle at top left, #0f172a 0, #020617 55%, #020617 100%);
            }
            /* Reduce top padding */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2.5rem;
                max-width: 900px;
            }
            /* Simple card look */
            .card {
                padding: 1.5rem 1.75rem;
                border-radius: 0.9rem;
                background: linear-gradient(145deg, #020617, #020617);
                border: 1px solid #1e293b;
                box-shadow: 0 18px 40px -24px rgba(15, 23, 42, 0.9);
            }
            .big-title {
                font-size: 2rem;
                font-weight: 600;
                margin-bottom: 0.3rem;
                color: #e5e7eb;
            }
            .subtitle {
                color: #9ca3af;
                font-size: 0.95rem;
                margin-bottom: 1.25rem;
            }
            /* Softer buttons */
            .stButton>button {
                border-radius: 999px;
                border: 1px solid #1d4ed8;
                padding: 0.5rem 1.4rem;
                font-weight: 500;
                background: linear-gradient(135deg, #1d4ed8, #38bdf8);
                color: #f9fafb;
            }
            .stDownloadButton>button {
                border-radius: 999px;
                padding: 0.4rem 1.1rem;
                border: 1px solid #1f2937;
                background: #020617;
                color: #e5e7eb;
            }
            .sidebar .sidebar-content, [data-testid="stSidebar"] {
                background: #020617;
            }
            .stSelectbox label, .stTextInput label {
                color: #e5e7eb !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.sidebar.markdown("### 🧠 Face Attendance")
        st.sidebar.caption("Minimal Streamlit demo")

        # Session / course metadata controls
        st.sidebar.markdown("#### Session settings")
        # Group theory first, then lab courses together
        course_options = [
            "CS231",
            "CS221",
            "ES205",
            "CE221",
            "CS202",
            "AI201 LAB",
            "CS221 LAB",
            "CE221 LAB",
        ]
        default_course = st.session_state.get("course", course_options[0])
        if default_course not in course_options:
            default_course = course_options[0]
        st.session_state["course"] = st.sidebar.selectbox(
            "Course",
            options=course_options,
            index=course_options.index(default_course),
        )
        st.session_state["session"] = st.sidebar.selectbox(
            "Session",
            options=["Morning", "Evening"],
            index=0 if st.session_state.get("session", "Morning") == "Morning" else 1,
        )

        page = st.sidebar.radio(
            "Navigate",
            options=["Home", "Register", "Mark Attendance", "View Attendance"],
            index=0,
        )

        if page == "Home":
            self.render_home()
        elif page == "Register":
            self.render_register()
        elif page == "Mark Attendance":
            self.render_attendance()
        else:
            self.render_history()


if __name__ == "__main__" or True:
    app = FaceAttendanceApp()
    app.run()
