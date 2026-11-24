import Foundation

// Entry point: headless CLI when `--render` is present, otherwise the SwiftUI app.
let cliArgs = Array(CommandLine.arguments.dropFirst())
if cliArgs.contains("--render") || cliArgs.contains("--help") {
    exit(Headless.run(cliArgs))
}
StudioApp.main()
