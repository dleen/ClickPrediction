name := "ClickPrediction Leen"

version := "1.0"

scalacOptions ++= Seq("-deprecation", "-unchecked")

libraryDependencies  ++= Seq(
            "colt" % "colt" % "1.2.0",
            "org.scalanlp" %% "breeze-math" % "0.1"
		)

resolvers ++= Seq(
            // other resolvers here
            // if you want to use snapshot builds (currently 0.2-SNAPSHOT), use this.
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
            "Sonatype Releases"  at "http://oss.sonatype.org/content/repositories/releases"
            )

mainClass in (Compile, run) := Some("main.scala.sgd.SGD")