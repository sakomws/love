const path = require("path");
const fs = require("fs");
const { exec } = require("child_process");

(function loadEnv(envFilePath = ".env") {
	const fullPath = path.resolve(process.cwd(), envFilePath);

	if (!fs.existsSync(fullPath)) {
		console.error(`.env file not found at path: ${fullPath}`);
		process.exit(1);
	}

	const envFileContent = fs.readFileSync(fullPath, "utf-8");

	for (const line of envFileContent.split("\n")) {
		const [key, value] = line.split("=").map((part) => part.trim());

		if (key && value !== undefined) {
			process.env[key] = value;
		}
	}
})();

const [inputFile, outputFile] = process.argv.slice(2);
const GROQ_API_KEY = process.env.GROQ_API_KEY;

if (!inputFile || !outputFile) {
	console.error(`Usage: node transcribe-audio <inputFile> <outputFile>

    It transcribes the audio file input.m4a using the Groq API.
    Example: node transcribe-audio input.m4a output.json`);
	process.exit(1);
}

async function executeCommand(command) {
	return new Promise((resolve, reject) => {
		exec(command, (error, stdout, stderr) => {
			if (error) {
				reject(`Error: ${error.message}`);
				return;
			}
			if (stderr) {
				// reject(`Stderr: ${stderr}`);
				resolve(stdout);
				return;
			}
			resolve(stdout);
		});
	});
}

async function transcribe(file) {
	const command = `
curl "https://api.groq.com/openai/v1/audio/transcriptions" \
  -H "Authorization: Bearer ${GROQ_API_KEY}" \
  -F "model=whisper-large-v3" \
  -F "file=@${file}" \
  -F "response_format=verbose_json" \
  -X POST
  `;
	console.log(`EXEC: ${command}`);
	return await executeCommand(command);
}

(async () => {
	const resp = await transcribe(inputFile);
	const filename = path.basename(inputFile);
	//write to outputDir
	fs.writeFileSync(outputFile, resp);
})();
