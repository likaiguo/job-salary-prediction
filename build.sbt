name := "salary"

version := "1.0"

scalaVersion := "2.10.0"

libraryDependencies ++= Seq(
  "org.mongodb" %% "casbah" % "2.5.0",
  "org.slf4j" % "slf4j-api" % "1.7.2",
  "org.slf4j" % "slf4j-simple" % "1.7.2",
  "net.sf.opencsv" % "opencsv" % "2.3"
)

mainClass in (Compile, packageBin) := Some("salary.SalaryMain")

mainClass in (Compile, run) := Some("salary.SalaryMain")
