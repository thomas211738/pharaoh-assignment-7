from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
import json  # Add this import
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t

app = Flask(__name__)
app.secret_key = "thomas211738"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # TODO 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    error = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + error

    # TODO 3: Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # TODO 4: Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data points')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Scatter plot with fitted regression line')
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # TODO 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N)
        error_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + error_sim

        # TODO 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

        # TODO 8: Plot histograms of slopes and intercepts
        plt.figure()
        plt.hist(slopes, bins=30, alpha=0.5, label='Slopes')
        plt.hist(intercepts, bins=30, alpha=0.5, label='Intercepts')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Histograms of slopes and intercepts')
        plot2_path = "static/plot2.png"
        plt.savefig(plot2_path)
        plt.close()

        # TODO 9: Return data needed for further analysis, including slopes and intercepts
        # Calculate proportions of slopes and intercepts more extreme than observed
        slope_more_extreme = np.mean(np.abs(slopes) > np.abs(slope))
        intercept_extreme = np.mean(np.abs(intercepts) > np.abs(intercept))

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = json.dumps(X.tolist())
        session["Y"] = json.dumps(Y.tolist())
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = json.dumps(slopes)
        session["intercepts"] = json.dumps(intercepts)
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = np.array(json.loads(session.get("slopes")), dtype=float)
    intercepts = np.array(json.loads(session.get("intercepts")), dtype=float)
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # TODO 10: Calculate p-value based on test type
    if test_type == "two-tailed":
        p_value = 2 * np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))
    elif test_type == "greater":
        p_value = np.mean(simulated_stats >= observed_stat)
    else:  # "less"
        p_value = np.mean(simulated_stats <= observed_stat)

    # TODO 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    if p_value <= 0.0001:
        fun_message = "Wow! That's a very small p-value!"
    else:
        fun_message = None

    # TODO 12: Plot histogram of simulated statistics
    plt.figure()
    plt.hist(simulated_stats, bins=30, alpha=0.5, label='Simulated stats')
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label=f'Observed {parameter}')
    plt.axvline(hypothesized_value, color='green', linestyle='dashed', linewidth=2, label=f'Hypothesized {parameter}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram of simulated statistics')
    
    plt.savefig('static/plot3.png')
    unique_id = np.random.randint(0, 100000)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3='static/plot3.png',
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        # TODO 13: Uncomment the following lines when implemented
        p_value=p_value,
        fun_message=fun_message,
        unique_id=unique_id,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))

    
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    
    slopes = np.array(json.loads(session.get("slopes")), dtype=float)
    intercepts = np.array(json.loads(session.get("intercepts")), dtype=float)

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))/100

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # TODO 14: Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)

    # TODO 15: Calculate confidence interval for the parameter estimate
    # Use the t-distribution and confidence_level
    print(confidence_level)
    t_value = t.ppf((1 + confidence_level) / 2, df=S-1)

    margin_of_error = t_value * (std_estimate / np.sqrt(S))
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error

    # TODO 16: Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # TODO 17: Plot the individual estimates as gray points and confidence interval
    # Plot the mean estimate as a colored point which changes if the true parameter is included
    # Plot the confidence interval as a horizontal line
    # Plot the true parameter value
    plt.figure()

    # Scatter plot for the simulated estimates
    plt.scatter(estimates, [0]*len(estimates), color='gray', alpha=0.5, label='Simulated Estimates')

    # Mean estimate with confidence interval error bar
    plt.errorbar(mean_estimate, 0, xerr=margin_of_error, fmt='o', color='blue', label='Mean Estimate')

    # Confidence interval
    plt.hlines(0, mean_estimate - margin_of_error, mean_estimate + margin_of_error, color='blue', linewidth=2, label=f'{confidence_level*100}% Confidence Interval')

    # True slope line
    plt.axvline(x=true_param, color='green', linestyle='dashed', linewidth=2, label=f'True {parameter}')

    plt.xlabel('Slope Estimate')
    plt.ylabel('')
    plt.yticks([])
    plt.title('90.0% Confidence Interval for Slope (Mean Estimate)')
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
