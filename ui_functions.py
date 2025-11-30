import streamlit as st
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import os
import face_recognition
# Import dependencies from other modules
from database import cursor, conn, load_lectures  
from face_utils import Images, classnames, encodeListKnown, load_known_faces, find_encodings
from emotion_utils import process_frame
from utils import get_current_lecture, dates

def add_new_face():
    st.subheader("Add New Face")
    new_name = st.text_input("Enter your name:")
    # 'step=1' ensures they can only enter whole numbers (integers)
# 'format="%d"' ensures it displays as 101, not 101.00
    roll_no_input = st.number_input("Enter your roll number:", min_value=1, step=1, format="%d")

# Convert back to string because filenames (Photo_101.jpg) must be strings
    roll_no = str(roll_no_input)

    # Camera controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera"):
            st.session_state.camera_active = True
    with col2:
        if st.button("Stop Camera"):
            st.session_state.camera_active = False

    if st.session_state.camera_active:
        img_file_buffer = st.camera_input("Take a picture")
        
        if img_file_buffer is not None:
            if not new_name or not roll_no:
                st.error("Please provide both Name and Roll Number.")
            else:
                # -----------------------------------------------------------
                # 🔍 FIX: Check Photos folder because DB isn't written to yet
                # -----------------------------------------------------------
                is_duplicate = False
                registered_name = ""

                # Scan the 'Photos' folder for this Roll No
                for filename in os.listdir("Photos"):
                    # Filename format is: Name_RollNo.jpg
                    if "_" in filename:
                        name_part = filename.split("_")[0]
                        roll_part = filename.split("_")[1].split(".")[0]
                        
                        if roll_part == roll_no:
                            is_duplicate = True
                            registered_name = name_part
                            break
                
                if is_duplicate:
                    st.error(f"⚠️ Roll Number {roll_no} is already registered to '{registered_name}'.")
                    st.session_state.camera_active = False
                else:
                    # Save the new face
                    image = np.array(Image.open(img_file_buffer))
                    img_path = os.path.join("Photos", f"{new_name}_{roll_no}.jpg")
                    cv2.imwrite(img_path, image)

                    # Refresh the recognition list immediately
                    global Images, classnames, encodeListKnown
                    Images, classnames = load_known_faces()
                    encodeListKnown = find_encodings(Images)

                    st.success(f"✅ Successfully registered {new_name} ({roll_no})!")
                    st.session_state.camera_active = False
    else:
        st.info("Camera is not active. Start the camera to take a picture.")
# --- Recognize Face and Emotion ---
def recognize_face():
    st.subheader("Recognize Face & Emotion (Mark Attendance)")
    lecture_schedule, faculty_schedule = load_lectures()
    
    current_lecture = get_current_lecture()
    current_day = datetime.now().strftime('%A')
    if current_lecture:
        st.info(f"Currently, {current_lecture} ({lecture_schedule[current_day][current_lecture]}) - {faculty_schedule[current_day][current_lecture]} is going on. You can mark attendance for this lecture only.")
        lecture = current_lecture  # Force selection to current lecture
    else:
        st.warning("No lecture is currently ongoing. Attendance cannot be marked.")
        return
    
    # Camera controls only in this mode
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera"):
            st.session_state.camera_active = True
    with col2:
        if st.button("Stop Camera"):
            st.session_state.camera_active = False

    if st.session_state.camera_active:
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            st.session_state.camera_active = False  # Stop camera after photo is taken
            with st.spinner("Processing..."):
                current_date = datetime.now().strftime('%Y-%m-%d')
                current_time = datetime.now().strftime('%H:%M:%S')
                scheduled_time = lecture_schedule[current_day][lecture]
                
                # Check if current date is in valid dates (excludes Sundays, 2nd/4th Saturdays, public holidays)
                if current_date in dates:
                    scheduled_dt = datetime.strptime(scheduled_time, '%H:%M')
                    current_dt = datetime.strptime(current_time[:5], '%H:%M')  # Use HH:MM for comparison
                    time_diff = abs((current_dt - scheduled_dt).total_seconds() / 60)
                    
                    if time_diff <= 10:  # Allow 10-minute window; no attendance outside this or free periods
                        image = np.array(Image.open(img_file_buffer))
                        imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
                        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                        facesCurFrame = face_recognition.face_locations(imgS)
                        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                        # Check if there are any known faces loaded
                        if not encodeListKnown:
                            st.error("⚠️ No registered faces found! Please go to 'Add New Face' and register a student first.")
                            return
                       
                        detected_emotions = []
                        if len(encodesCurFrame) > 0:
                            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                                matchIndex = np.argmin(faceDis)

                                if matches[matchIndex]:
                                    name = classnames[matchIndex].split("_")[0]
                                    roll_no = classnames[matchIndex].split("_")[1]

                                    y1, x2, y2, x1 = faceLoc
                                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                                    frame, detected_emotions = process_frame(image)
                                    emotion = detected_emotions[0] if detected_emotions else "Unknown"

                                    cursor.execute("INSERT INTO attendance (name, roll_no, date, time, lecture, status, emotion) VALUES (?, ?, ?, ?, ?, 'Present', ?)", 
                                                   (name, roll_no, current_date, current_time, lecture, emotion))
                                    conn.commit()
                                    st.success(f"Attendance marked for {name} in {lecture} (Faculty: {faculty_schedule[current_day][lecture]}) with emotion: {emotion}.")
                                else:
                                    st.warning("Face not recognized.")
                        else:
                            st.warning("No face detected.")
                        st.image(image, caption="Detected Face and Emotion", use_container_width=True)
                    else:
                        st.error(f"Attendance can only be marked within 10 minutes of the scheduled time for {lecture} ({scheduled_time}). No attendance in free lectures.")
                else:
                    st.error("Attendance can only be marked on valid dates (weekdays, allowed Saturdays, excluding public holidays).")
    else:
        st.info("Camera is not active. Start the camera to take a picture.")

# --- View Timetable ---
def view_timetable():
    st.subheader("Weekly Lecture Timetable")
    
    # RELOAD DATA: Fetch the latest schedule from DB
    lecture_schedule, faculty_schedule = load_lectures()

    timetable_data = []
    for day, lectures in lecture_schedule.items():
        for lec, time in lectures.items():
            timetable_data.append({
                "Day": day,
                "Time": time,
                "Lecture": f"{lec} ({faculty_schedule[day][lec]})"
            })
    
    if not timetable_data:
        st.info("No lectures scheduled yet. Please go to the Admin Panel to add Faculty and Lectures.")
        return

    df_timetable = pd.DataFrame(timetable_data)
    # Pivot the table: Rows = Days, Columns = Times, Values = Lectures
    pivot_df = df_timetable.pivot_table(index="Day", columns="Time", values="Lecture", aggfunc='first').fillna('')
    st.table(pivot_df)


# --- Close Attendance for Lecture (Mark Absences) ---
def close_attendance():
    # RELOAD DATA: Load fresh data to avoid NameError and get latest lecture info
    lecture_schedule, faculty_schedule = load_lectures()

    st.subheader("Close Attendance for Lecture (Mark Absences)")
    
    current_lecture = get_current_lecture()
    current_day = datetime.now().strftime('%A')
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')
    
    if current_lecture:
        scheduled_time = lecture_schedule[current_day][current_lecture]
        scheduled_dt = datetime.strptime(scheduled_time, '%H:%M')
        current_dt = datetime.strptime(current_time, '%H:%M')
        time_diff = (current_dt - scheduled_dt).total_seconds() / 60
        
        st.info(f"Currently, {current_lecture} ({scheduled_time}) - {faculty_schedule[current_day][current_lecture]} is going on.")
        
        # 1. Automatic Closing (Time > 15 mins)
        if time_diff > 15:
            st.warning("15 minutes have passed. Automatically closing attendance.")
            
            registered_students = [(cls.split("_")[0], cls.split("_")[1]) for cls in classnames]
            
            # FIX: Check for ANY status (Present or Absent) to avoid duplicates
            cursor.execute("SELECT name, roll_no FROM attendance WHERE date = ? AND lecture = ?", (current_date, current_lecture))
            already_marked = set((row[0], row[1]) for row in cursor.fetchall())
            
            absences = []
            for name, roll_no in registered_students:
                if (name, roll_no) not in already_marked:
                    cursor.execute("INSERT INTO attendance (name, roll_no, date, time, lecture, status, emotion) VALUES (?, ?, ?, ?, ?, 'Absent', 'N/A')", 
                                   (name, roll_no, current_date, scheduled_time, current_lecture))
                    absences.append(f"{name} ({roll_no})")
            conn.commit()
            
            if absences:
                st.success(f"Automatically marked absent for: {', '.join(absences)}")
            else:
                st.info("Attendance already closed or everyone is present.")

        # 2. Manual Closing Button
        else:
            if st.button("Close Attendance Now"):
                registered_students = [(cls.split("_")[0], cls.split("_")[1]) for cls in classnames]
                
                # FIX: Check for ANY status (Present or Absent)
                cursor.execute("SELECT name, roll_no FROM attendance WHERE date = ? AND lecture = ?", (current_date, current_lecture))
                already_marked = set((row[0], row[1]) for row in cursor.fetchall())
                
                absences = []
                for name, roll_no in registered_students:
                    if (name, roll_no) not in already_marked:
                        cursor.execute("INSERT INTO attendance (name, roll_no, date, time, lecture, status, emotion) VALUES (?, ?, ?, ?, ?, 'Absent', 'N/A')", 
                                       (name, roll_no, current_date, scheduled_time, current_lecture))
                        absences.append(f"{name} ({roll_no})")
                conn.commit()
                
                if absences:
                    st.success(f"Marked absent for: {', '.join(absences)}")
                else:
                    st.info("Attendance already closed or everyone is present.")
    else:
        st.warning("No lecture is currently ongoing. Attendance cannot be closed.")
# --- View Attendance Records (Pivoted with Lectures as Columns + Detailed Table + Export + Person Selection) ---
def view_attendance_records():
    
    lecture_schedule, faculty_schedule = load_lectures()
    
    st.subheader("Attendance Records (Lectures as Columns)")
   
    # Calendar date selection
    min_date = datetime.strptime(min(dates), '%Y-%m-%d').date()
    max_date = datetime.strptime(max(dates), '%Y-%m-%d').date()
    selected_date_obj = st.date_input("Select Date to View", value=datetime.now().date(), min_value=min_date, max_value=max_date)
    selected_date = selected_date_obj.strftime('%Y-%m-%d')
    
    if selected_date not in dates:
        st.error("Selected date is not a valid date (e.g., Sunday, excluded Saturday, or holiday). Please select a valid date.")
        return
    
    selected_day = selected_date_obj.strftime('%A')
    if selected_day not in lecture_schedule:
        st.error("No lectures on selected date.")
        return
    
    # Fetch records for the selected date
    cursor.execute("SELECT name, roll_no, lecture, status FROM attendance WHERE date = ?", (selected_date,))
    records = cursor.fetchall()
    
    if records:
        df = pd.DataFrame(records, columns=["Name", "Roll No", "Lecture", "Status"])
        
        # Pivoted Table: Rows = Students, Columns = Lectures, Values = Status
        pivot_df = df.pivot_table(index=["Name", "Roll No"], columns="Lecture", values="Status", aggfunc='first').reset_index()
        pivot_df = pivot_df.fillna('Not Marked')
        
        # Update lecture column names to include time (e.g., "l1(09:00)")
        for lec in lecture_schedule[selected_day]:
            if lec in pivot_df.columns:
                pivot_df.rename(columns={lec: f"{lec}({lecture_schedule[selected_day][lec]})"}, inplace=True)
        
        # Add percentages (adjusted for valid dates and weekly lectures)
        total_lectures = len(lecture_schedule[selected_day])
        total_valid_dates = len(dates)
        
        percentages = []
        for _, row in pivot_df.iterrows():
            name, roll_no = row["Name"], row["Roll No"]
            daily_present = sum(1 for lec in lecture_schedule[selected_day] if row.get(f"{lec}({lecture_schedule[selected_day][lec]})", 'Not Marked') == 'Present')
            daily_percentage = (daily_present / total_lectures) * 100 if total_lectures > 0 else 0
            
            cursor.execute("SELECT COUNT(*) FROM attendance WHERE name = ? AND roll_no = ? AND status = 'Present'", (name, roll_no))
            overall_present = cursor.fetchone()[0]
            overall_possible = total_valid_dates * total_lectures  # Approximate, as lectures vary by day
            overall_percentage = (overall_present / overall_possible) * 100 if overall_possible > 0 else 0
            
            percentages.append({
                "Daily Attendance %": f"{daily_percentage:.2f}%",
                "Overall Attendance %": f"{overall_percentage:.2f}%"
            })
        
       # ... inside view_attendance_records ...

        percentages_df = pd.DataFrame(percentages)
        final_pivot_df = pd.concat([pivot_df, percentages_df], axis=1)
        
       
        final_pivot_df["Roll No"] = pd.to_numeric(final_pivot_df["Roll No"], errors='ignore')
        final_pivot_df = final_pivot_df.sort_values(by="Roll No")
        
        st.table(final_pivot_df)
       
        st.subheader("Lecture-Wise Status (Pivoted)")
        st.table(final_pivot_df)
        
        # Detailed Table: All records with Time and Emotion
        cursor.execute("SELECT name, roll_no, lecture, status, time, emotion FROM attendance WHERE date = ? ORDER BY roll_no, time DESC", (selected_date,))
        detailed_records = cursor.fetchall()
        if detailed_records:
            detailed_df = pd.DataFrame(detailed_records, columns=["Name", "Roll No", "Lecture", "Status", "Time", "Emotion"])
            # Update "Lecture" column to include time
            detailed_df["Lecture"] = detailed_df["Lecture"].map(lambda x: f"{x}({lecture_schedule[selected_day].get(x, '')})" if x in lecture_schedule[selected_day] else x)
            st.subheader("Detailed Records (with Time and Emotion)")
            st.table(detailed_df)
            
            # Export to CSV
            csv = detailed_df.to_csv(index=False)
            st.download_button(
                label="Download Detailed Records as CSV",
                data=csv,
                file_name=f"attendance_records_{selected_date}.csv",
                mime="text/csv",
                key="download_csv"
            )
        else:
            st.info("No detailed records available.")
        
        # Add selectbox for selecting a person to view their full details (click to expand below)
        person_options = [f"{row['Name']} ({row['Roll No']})" for _, row in final_pivot_df.iterrows()]
        selected_person = st.selectbox("Select a person to view their full details", [""] + person_options)
        
        if selected_person:
            name, roll_no = selected_person.split(" (")
            roll_no = roll_no.rstrip(")")
            st.session_state.selected_person = (name, roll_no)
            
            st.markdown("---")
            st.subheader(f"Full Details for {name} (Roll No: {roll_no})")
            
            # --- NEW: Calculate & Display Percentage Metric ---
            cursor.execute("SELECT COUNT(*) FROM attendance WHERE name = ? AND roll_no = ? AND status = 'Present'", (name, roll_no))
            total_present = cursor.fetchone()[0]
            
            avg_lectures_per_day = sum(len(v) for v in lecture_schedule.values()) / len(lecture_schedule) if lecture_schedule else 0
            total_possible = len(dates) * avg_lectures_per_day
            user_percentage = (total_present / total_possible * 100) if total_possible > 0 else 0.0
            
            # Show Big Metric
            st.metric(label="Overall Attendance Percentage", value=f"{user_percentage:.2f}%")
            
            # Fetch detailed logs
            cursor.execute("SELECT date, lecture, status, time, emotion FROM attendance WHERE name = ? AND roll_no = ? ORDER BY date DESC, time DESC", (name, roll_no))
            person_records = cursor.fetchall()
            
            if person_records:
                df_person = pd.DataFrame(person_records, columns=["Date", "Lecture", "Status", "Time", "Emotion"])
                
                # Format lecture names
                df_person["Lecture"] = df_person.apply(
                    lambda row: f"{row['Lecture']}({lecture_schedule.get(datetime.strptime(row['Date'], '%Y-%m-%d').strftime('%A'), {}).get(row['Lecture'], '')})" 
                    if row['Lecture'] in lecture_schedule.get(datetime.strptime(row['Date'], '%Y-%m-%d').strftime('%A'), {}) 
                    else row['Lecture'], 
                    axis=1
                )
                
                # --- NEW: Add Percentage Column to the DataFrame ---
                df_person["Current Overall %"] = f"{user_percentage:.2f}%"
                
                st.table(df_person)
                
                # Export Button
                csv_person = df_person.to_csv(index=False)
                st.download_button(
                    label=f"Download Records for {name}",
                    data=csv_person,
                    file_name=f"{name}_{roll_no}_records.csv",
                    mime="text/csv",
                    key="download_person_csv"
                )
            else:
                st.info("No records found for this person.")
    else:
        st.info(f"No attendance records available for {selected_date}.")
# --- View Attendance Summary (Percentages Only) ---
def view_attendance_summary():
 
    lecture_schedule, faculty_schedule = load_lectures()

    st.subheader("Attendance Summary and Percentages")
    
    cursor.execute("SELECT DISTINCT name, roll_no FROM attendance")
    students = cursor.fetchall()
    
    if not students:
        st.info("No attendance records available to calculate percentages.")
        return
    
    summary_data = []
    current_date = datetime.now().strftime('%Y-%m-%d')
    total_dates = len(dates)
    # Approximate total lectures per day (average across days)
    total_lectures_per_day = sum(len(lectures) for lectures in lecture_schedule.values()) / len(lecture_schedule)
    
    for name, roll_no in students:
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE name = ? AND roll_no = ? AND date = ? AND status = 'Present'", (name, roll_no, current_date))
        daily_present = cursor.fetchone()[0]
        daily_percentage = (daily_present / total_lectures_per_day) * 100 if total_lectures_per_day > 0 else 0
        
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE name = ? AND roll_no = ? AND status = 'Present'", (name, roll_no))
        overall_present = cursor.fetchone()[0]
        overall_possible = total_dates * total_lectures_per_day
        overall_percentage = (overall_present / overall_possible) * 100 if overall_possible > 0 else 0
        
        summary_data.append({
            "Name": name,
            "Roll No": roll_no,
            "Daily Attendance % (Today)": f"{daily_percentage:.2f}%",
            "Overall Attendance %": f"{overall_percentage:.2f}%"
        })
    

    
    df_summary = pd.DataFrame(summary_data)
    df_summary["Roll No"] = pd.to_numeric(df_summary["Roll No"], errors='ignore')
    df_summary = df_summary.sort_values(by="Roll No") 
    
    st.table(df_summary)
   
    
    # Export to CSV
    csv = df_summary.to_csv(index=False)
    st.download_button(
        label="Download Attendance Records as CSV",
        data=csv,
        file_name=f"attendance_records.csv",
        mime="text/csv",
        key="download_csv",
        use_container_width=True,
        type='primary'
    )
    st.info("Table is sorted by Roll No. Percentages update automatically based on current records.")
conn.commit()
# --- Initialize session_state attributes ---
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "selected_person" not in st.session_state:
    st.session_state.selected_person = None
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
