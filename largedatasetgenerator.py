def generate_grid(rows, cols, filename):
    with open(filename, "w") as out:

        # Coordinates section
        out.write("# Coordinates\n")
        for r in range(rows):
            for c in range(cols):
                # Format: r0c0 x y
                out.write(f"r{r}c{c} {c} {r}\n")

        out.write("\n")  

        # Edges section
        out.write("# Edges\n")
        for r in range(rows):
            for c in range(cols):
                name = f"r{r}c{c}"

                # Edge to the right
                if c + 1 < cols:
                    out.write(f"{name} r{r}c{c+1} 1\n")

                # Edge downward
                if r + 1 < rows:
                    out.write(f"{name} r{r+1}c{c} 1\n")


if __name__ == "__main__":
    try:
        rows = int(input("Enter number of rows: "))
        cols = int(input("Enter number of columns: "))
        filename = input("Enter output filename (e.g., demo3.txt): ")

        generate_grid(rows, cols, filename)
        print(f"\nFile '{filename}' successfully generated with {rows * cols} nodes.")

    except ValueError:
        print("Error: Please enter valid integer values for rows and columns.")
