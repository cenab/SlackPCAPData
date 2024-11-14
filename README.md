# Slack Network Activity Analyzer

## Introduction

This project is designed to simulate and analyze network activity associated with the Slack application. It includes scripts that iterate through lines from **Hamlet**, decide which Slack channel to post to, and simulate user interactions within the Slack app. The project also provides tools for capturing and analyzing network traffic, including packet captures and netstat outputs.

## Features

- **Simulate Slack Activity**: Automatically post messages to Slack channels, simulating user activity.
- **Network Traffic Capture**: Capture and analyze network packets related to Slack activity.
- **Flowchart of Application Logic**: A visual representation of the application's logic flow.
- **Network Analysis Tools**: Scripts and commands to analyze network connections and packet captures.

## Installation

### Prerequisites

- Install [Docker](https://docs.docker.com/get-docker/) on your system.

### Clone the Repository

```bash
git clone https://github.com/yourusername/slack-network-activity-analyzer.git
cd slack-network-activity-analyzer
```

### Building the Docker Image

Build the Docker image using the provided `Dockerfile`:

```bash
docker build -t slack-analyzer .
```

## Usage

### Running the Application

Run the Docker container:

```bash
docker run -it --rm slack-analyzer
```

This will start the simulation of Slack activity and begin data collection.

### Capturing Network Traffic

The application includes scripts to capture network traffic using `tcpdump`:

```bash
bash tcpdump_files/start_tcpdump.sh
```

Ensure that `tcpdump` is properly set up and has the necessary permissions.

### Collecting Netstat Outputs

To collect network connection information over time:

```bash
bash tcpdump_files/netstat_collect_ports.sh
```

This script will log netstat outputs to `netstat_output.txt` at regular intervals.

### Data Analysis

After running the simulations and capturing data, you can analyze the network traffic using provided scripts in the `ml_analysis` directory.

## Project Structure

- `start_data_collection.py`: Main script to start data collection and simulate Slack activity.
- `user_interaction/`: Contains scripts and functions to interact with the Slack application.
- `tcpdump_files/`: Contains scripts for capturing network traffic.
- `ml_analysis/`: Contains scripts for machine learning analysis of the collected data.
- `Dockerfile`: Dockerfile to containerize the application.
- `requirements.txt`: Python dependencies for the project.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `.gitattributes`: Configures Git LFS for large files like `.pcap` files.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

---

## Making the Project Git Presentable

To make your project more presentable and user-friendly on GitHub:

1. **Organize the Repository**: Ensure that your project directories are logically organized, and scripts are placed in appropriate folders.

2. **Update `.gitignore`**: Include a `.gitignore` file to prevent tracking of unnecessary files like temporary files, compiled binaries, and large datasets.

   ```gitignore:.gitignore
   # Byte-compiled / optimized / DLL files
   __pycache__/
   *.pyc
   *.pyo
   *.pyd

   # Virtual environment
   venv/
   env/

   # Environment variables
   .env

   # MacOS specific files
   .DS_Store

   # Ignore traceintel database and large pcap files
   traceintel.db
   /pcap_edited/
   *.pcap
   *.pcapng
   ```

3. **Use `.gitattributes` for Large Files**: Configure Git LFS for handling large files like packet captures.

   ```gitattributes:.gitattributes
   *.pcap filter=lfs diff=lfs merge=lfs -text
   *.pcapng filter=lfs diff=lfs merge=lfs -text
   ```

4. **Provide Clear Documentation**:

   - **README.md**: Enhance your README with detailed instructions, examples, and explanations.
   - **Comments**: Add comments to your code to explain complex sections.
   - **Docstrings**: Use docstrings in your Python functions and modules.

5. **Include a License**: Add a LICENSE file to specify the project's license.

6. **Add a Contribution Guide**: If you expect others to contribute, include a `CONTRIBUTING.md` file with guidelines.

7. **Use Descriptive Commit Messages**: Write clear and concise commit messages that describe the changes made.

8. **Provide Examples and Tests**:

   - Include example data or usage examples.
   - Add tests to ensure your code works as expected.

9. **Clean Up the Code**:

   - Remove any unnecessary code or files.
   - Refactor code for readability and efficiency.

10. **Screenshots and Visuals**: If applicable, add screenshots or diagrams to help users understand the project.

---

By following these steps, you'll make your project more accessible and appealing to other users and contributors.

Feel free to reach out if you need further assistance!
