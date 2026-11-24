import Foundation

// Entry point: headless CLI for --render / --receive / --reconstruct, otherwise the SwiftUI app.
let cliArgs = Array(CommandLine.arguments.dropFirst())
if cliArgs.contains("--receive") {
    exit(HeadlessLive.receive(cliArgs))
}
if cliArgs.contains("--reconstruct") {
    exit(HeadlessLive.reconstruct(cliArgs))
}
if cliArgs.contains("--render") || cliArgs.contains("--help") {
    if cliArgs.contains("--help") { print(HeadlessLive.usage) }
    exit(Headless.run(cliArgs))
}
StudioApp.main()
