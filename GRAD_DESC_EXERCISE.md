# Gradient Descent Exercise - Easy Instructions

## What is this exercise about?

In this exercise, you will learn how to use gradient descent to find the best answer.
There are two ways you will try:

1. **Using only the function:**
   - You can ask the computer only for the value of the function at different points.
   - You will find the slope (gradient) by testing small changes in the answer (this is called using a small epsilon).
   - This way is slow because you need to ask for many points.

2. **Using the function and its slope (gradient):**
   - You can ask the computer for both the answer and the slope.
   - This way is faster and better.

## What should you do?

- Write code to do gradient descent with both ways.
- Count how many times you ask for the function and slope.
- Count the basic math operations (additions, multiplications, divisions) that you do.
- See which way is faster and gives a better answer.

## Rules

- You cannot see inside the code of the function.
- You can only use the API to get answers from the function or its slope.

## Details about the math

- To find the slope with only the function, use a small number (epsilon), similar to what we did in a previous exercise.
- Each slope part needs two function answers: one with a little more, one with a little less.
- Keep track of how many times you get answers from the function and slope.
- Also keep track of how many math operations you do.

## What you need to turn in

- Code for gradient descent both ways.
-- A report or log showing:
   - How many steps you did.
   - How many function answers you got.
   - How many gradient answers you got.
   - How many math operations you performed.
   - Graphs that show how well you did.
   - Present your results as dataframes using `display(dataframe)` to make automatic checking easier.
- A short explanation of which way is better and why.

## Why do this?

This exercise helps you learn about how computers find optimal solutions efficiently, especially in the context of neural network training and optimization.


# Gradient Descent Exercise - Easy Instructions

> Reminder: In this exercise, **slope** means the same as **gradient**. We will use the word **gradient** here.

> Reminder: An **analytic gradient** means the exact formula for the gradient of a function, if it can be calculated mathematically.<br/>
> For example, if $$f(x) = x^{2} - 10x$$ the analytic gradient is $$f'(x) = 2x - 10$$

## Exercise Steps

These steps guide you through calling the API, running simple function and gradient experiments, and counting calls and calculations.

1. **Explore the API for function evaluation**
   - Call the API to get the function value at a given point x.
   - Try this on simple 1D and 2D inputs.
   - Count how many times you call the API.

2. **Brush-up reminder: Numerical gradient estimation with finite differences**
   - Use a small epsilon to estimate gradient by checking function at x+epsilon and x-epsilon.
   - Write code to estimate the gradient by this method calling the API.
   - Test on 1D and 2D inputs.
   - Count API calls and math operations done in estimation.

3. **Use analytic gradient from the API**
   - Call the API to get analytic gradient directly.
   - Test on 1D and 2D inputs.
   - Count API calls and math operations.

4. **Build and run gradient descent**
   - Implement gradient descent optimizer using your numeric gradient function.
   - Run on 1D and 2D functions; count calls and math operations.
   - Repeat with analytic gradient.
   - Compare efficiency and solution quality.

5. **Summarize your findings**
   - Compare number of calls, operation counts, and accuracy between numeric and analytic gradients.
   - Discuss which method is faster and why.
