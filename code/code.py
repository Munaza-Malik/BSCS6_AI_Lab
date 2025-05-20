import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import threading, time, random
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ========== Meal Reminder ==========
def reminder_thread():
    while True:
        time.sleep(3600)
        messagebox.showinfo("Meal Reminder", "⏰ Time to drink water or have a healthy snack!")


threading.Thread(target=reminder_thread, daemon=True).start()

# ========== Dataset ==========
data = {
    'current_weight': [70, 80, 65, 90, 75, 60, 85, 72, 100, 95, 92, 105, 60, 55, 110, 115, 70, 65, 50, 98, 62, 77],
    'target_weight': [68, 75, 63, 80, 70, 55, 78, 68, 50, 60, 55, 70, 58, 52, 90, 95, 68, 60, 48, 85, 60, 70],
    'days': [7, 10, 7, 14, 10, 7, 12, 8, 30, 25, 40, 60, 15, 12, 40, 50, 20, 25, 10, 45, 18, 20],
    'activity': ['medium', 'high', 'low', 'high', 'medium', 'low', 'medium', 'low', 'low', 'medium', 'low', 'medium',
                 'low', 'medium', 'medium', 'high', 'medium', 'low', 'low', 'high', 'medium', 'high'],
    'diet': ['veg', 'mixed', 'veg', 'non-veg', 'mixed', 'veg', 'non-veg', 'veg', 'veg', 'mixed', 'veg', 'mixed',
             'mixed', 'veg', 'non-veg', 'non-veg', 'mixed', 'veg', 'veg', 'mixed', 'veg', 'non-veg'],
    'result': ['high', 'high', 'moderate', 'high', 'moderate', 'low', 'moderate', 'low', 'low', 'low', 'low',
               'moderate',
               'moderate', 'moderate', 'high', 'high', 'moderate', 'moderate', 'low', 'high', 'moderate', 'high']
}

df = pd.DataFrame(data)
le_activity, le_diet, le_result = LabelEncoder(), LabelEncoder(), LabelEncoder()
df['activity'] = le_activity.fit_transform(df['activity'])
df['diet'] = le_diet.fit_transform(df['diet'])
df['result'] = le_result.fit_transform(df['result'])
model = GaussianNB().fit(df[['current_weight', 'target_weight', 'days', 'activity', 'diet']], df['result'])


X = df[['current_weight', 'target_weight', 'days', 'activity', 'diet']]
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB().fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"📈 Model Accuracy on Test Set: {round(accuracy * 100, 2)}%")
# ========== Utility Functions ==========
activity_multiplier = {'low': 1.2, 'medium': 1.5, 'high': 1.8}
meal_plans = {
    'veg': {'breakfast': 'Oats + Fruits', 'lunch': 'Boiled lentils + veggies', 'dinner': 'Light salad'},
    'non-veg': {'breakfast': 'Boiled eggs + Toast', 'lunch': 'Grilled chicken + rice', 'dinner': 'Fish + Veggies'},
    'mixed': {'breakfast': 'Oats + Egg', 'lunch': 'Paneer/Chicken + Rice', 'dinner': 'Soup + Salad'}
}
health_tips = [
    "Drink 2L of water daily.", "Never skip breakfast.", "Sleep 7-8 hours.",
    "Avoid fast food.", "Consistency is key."
]


def get_random_tip(): return f"💡 Tip: {random.choice(health_tips)}"


def calculate_bmi(w, h): return round(w / (h / 100) ** 2, 1)


def interpret_bmi(
        bmi): return "Underweight 🟡" if bmi < 18.5 else "Normal ✅" if bmi < 25 else "Overweight 🟠" if bmi < 30 else "Obese 🔴"


def calculate_bmr(w, h, a, g): return int(
    10 * w + 6.25 * h - 5 * a + (5 if g == 'male' else -161 if g == 'female' else 0))


def check_allergies(meals, allergies):
    return [f"⚠ {meal}: contains '{allergy}'" for meal, items in meals.items() for allergy in allergies if
            allergy.lower() in items.lower()]


def generate_plan(cw, tw, d, act, diet, allergies, gender, age, height):
    if any(v <= 0 for v in [cw, tw, d, age, height]):
        return "⚠ Invalid input values."
    bmi = calculate_bmi(cw, height)
    bmr = calculate_bmr(cw, height, age, gender.lower())
    maintenance = int(bmr * activity_multiplier.get(act, 1.2))
    meals = meal_plans.get(diet, meal_plans['mixed'])
    act_time = 30 if act == 'low' else 45 if act == 'medium' else 60
    diff = tw - cw
    plan = {
        "BMI": f"{bmi} ({interpret_bmi(bmi)})", "Days": d,
        "Meals": meals, "Activity (min)": act_time,
        "Warnings": check_allergies(meals, allergies)
    }
    if diff == 0: return "🎯 Already at target weight!"
    cal_change = round(abs(diff) * 7700 / d)
    plan.update({
        "Goal": "Weight Gain" if diff > 0 else "Weight Loss",
        "Weight to Gain" if diff > 0 else "Weight to Lose": round(abs(diff), 2),
        "Calorie Surplus" if diff > 0 else "Calorie Deficit": cal_change,
        "Recommended Calories": maintenance + cal_change if diff > 0 else max(1000, maintenance - cal_change)
    })
    return plan


# ========== GUI Setup ==========
root = tk.Tk()
root.title("🌿 AI Diet Planner")
root.geometry("700x600")
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# ========== Tab 1: Plan Screen ==========
tab1 = tk.Frame(notebook, bg='#e2f8fb')
notebook.add(tab1, text='Diet Plan')

labels = ["Current Weight (kg)", "Target Weight (kg)", "Height (cm)", "Age", "Days to Goal"]
entries = {}
for i, lbl in enumerate(labels):
    tk.Label(tab1, text=lbl, bg='#e6f7ff', font=('Arial', 11, 'bold')).grid(row=i, column=0, padx=10, pady=5,
                                                                            sticky='e')
    entries[lbl] = tk.Entry(tab1, width=30);
    entries[lbl].grid(row=i, column=1, pady=5)

tk.Label(tab1, text="Gender", bg='#e6f7ff', font=('Arial', 11, 'bold')).grid(row=5, column=0, sticky='e')
gender_box = ttk.Combobox(tab1, values=["Male", "Female", "Other"]);
gender_box.current(0)
gender_box.grid(row=5, column=1)

tk.Label(tab1, text="Activity Level", bg='#e6f7ff', font=('Arial', 11, 'bold')).grid(row=6, column=0, sticky='e')
activity_box = ttk.Combobox(tab1, values=["low", "medium", "high"]);
activity_box.current(1)
activity_box.grid(row=6, column=1)

tk.Label(tab1, text="Diet Type", bg='#e6f7ff', font=('Arial', 11, 'bold')).grid(row=7, column=0, sticky='e')
diet_box = ttk.Combobox(tab1, values=["veg", "non-veg", "mixed"]);
diet_box.current(0)
diet_box.grid(row=7, column=1)

tk.Label(tab1, text="Allergies (comma-separated)", bg='#e6f7ff', font=('Arial', 11, 'bold')).grid(row=8, column=0,
                                                                                                  sticky='e')
allergy_entry = tk.Entry(tab1, width=30);
allergy_entry.grid(row=8, column=1)

output_box = tk.Text(tab1, height=12, width=70, bg='#f0ffff');
output_box.grid(row=10, column=0, columnspan=2, pady=10, sticky='ew')
tab1.grid_columnconfigure(0, weight=1)
tab1.grid_columnconfigure(1, weight=1)

print("Model accuracy:", accuracy_score(y_test, y_pred))

def on_submit():
    try:
        cw, tw = float(entries["Current Weight (kg)"].get()), float(entries["Target Weight (kg)"].get())
        height, age = float(entries["Height (cm)"].get()), int(entries["Age"].get())
        days = int(entries["Days to Goal"].get())
        gender, activity, diet = gender_box.get(), activity_box.get(), diet_box.get()
        allergies = [a.strip() for a in allergy_entry.get().split(',') if a.strip()]

        # Prediction logic
        pred_label = "low"
        if abs(tw - cw) / days < 1:
            pred_label = "moderate"
        if abs(tw - cw) / days < 0.5:
            pred_label = "high"

        output_box.config(state='normal')
        output_box.delete('1.0', tk.END)
        output_box.insert(tk.END,
                          f"👤 {gender}, {age} yrs | {height} cm\n📊 Success Chance: {pred_label.upper()}\n" + "-" * 40 + "\n")

        plan = generate_plan(cw, tw, days, activity, diet, allergies, gender, age, height)
        if isinstance(plan, str):
            output_box.insert(tk.END, plan)
        else:
            output_box.insert(tk.END, f"{'🏋' if plan['Goal'] == 'Weight Gain' else '🏃'} {plan['Goal']} Plan:\n")
            for k, v in plan.items():
                if isinstance(v, dict):
                    for mk, mv in v.items():
                        output_box.insert(tk.END, f"{mk}: {mv}\n")
                elif isinstance(v, list):
                    for item in v:
                        output_box.insert(tk.END, f"{item}\n")
                else:
                    output_box.insert(tk.END, f"{k}: {v}\n")
        output_box.insert(tk.END, "\n" + get_random_tip())
        output_box.config(state='disabled')
    except Exception as e:
        messagebox.showerror("Error", f"⚠ {e}")


tk.Button(tab1, text="Generate Plan", bg='purple', fg='white', font=('Arial', 11, 'bold'), command=on_submit).grid(
    row=9, column=0, pady=10)


def clear_all_fields():
    output_box.config(state='normal')
    output_box.delete('1.0', tk.END)
    for e in entries.values(): e.delete(0, tk.END)
    allergy_entry.delete(0, tk.END)
    gender_box.set("Male")
    activity_box.set("medium")
    diet_box.set("veg")
    output_box.config(state='disabled')


tk.Button(tab1, text="Clear Output", bg='#3b64b3', fg='white',
          font=('Arial', 11, 'bold'), command=clear_all_fields).grid(row=9, column=1)


# ========== Tab 2: Progress Tracker ==========
tab2 = tk.Frame(notebook, bg='#e2f8fb')
notebook.add(tab2, text='Progress Tracker')
progress_data = []

tk.Label(tab2, text="Today's Weight (kg):", font=('Arial', 11, 'bold')).pack(pady=5)
weight_entry = tk.Entry(tab2, width=30); weight_entry.pack()
tk.Label(tab2, text="Date (e.g. 2025-05-19):", font=('Arial', 11, 'bold')).pack(pady=5)
date_entry = tk.Entry(tab2, width=30); date_entry.pack()
def log_progress():
    try:
        weight = float(weight_entry.get())
        date = date_entry.get().strip()
        if not date: raise ValueError("Date cannot be empty.")
        progress_data.append((date, weight))
        messagebox.showinfo("Logged", f"✅ Progress saved for {date}")
    except Exception as e:
        messagebox.showerror("Error", f"⚠ {e}")
def show_progress():
    if not progress_data:
        messagebox.showinfo("No Data", "No progress to show.")
        return
    dates, weights = zip(*progress_data)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dates, weights, marker='o', color='blue')
    ax.set_title("Weight Progress")
    ax.set_xlabel("Date"); ax.set_ylabel("Weight (kg)"); ax.grid(True)
    top = tk.Toplevel(root); top.title("📈 Progress Chart")
    top.geometry("800x400")
    top.resizable(True, True)
    canvas = FigureCanvasTkAgg(fig, top); canvas.draw(); canvas.get_tk_widget().pack()
tk.Button(tab2, text="Log Progress", bg='#3b64b3', fg='white', font=('Arial', 10, 'bold'), command=log_progress).pack(pady=10)
tk.Button(tab2, text="View Progress Chart", bg='purple', fg='white', font=('Arial', 10, 'bold'), command=show_progress).pack(pady=2)
# ========== Start App ==========
root.mainloop()